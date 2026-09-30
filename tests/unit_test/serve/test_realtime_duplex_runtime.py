# SPDX-License-Identifier: Apache-2.0
"""Session runtime contracts that depend on scheduling the wire cannot pin."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager

import pytest

from sglang_omni.serve.realtime.control import Closed, Drained, Failure, UnitCompleted
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
)
from sglang_omni.serve.realtime.output_buffer import OutputBuffer
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import (
    Capabilities,
    Envelope,
    InteractionAdapter,
    OutputSink,
    ProtocolError,
    RuntimeLimits,
    Unit,
)

MODEL_NAME = "duplex-test"
SAMPLE_RATE = 16000
NATIVE_UNIT_MS = 20
UNIT_BYTES = SAMPLE_RATE * NATIVE_UNIT_MS // 1000 * 2
ACTIVITY_TIMEOUT_S = 0.5
TEST_WAIT_TIMEOUT_S = 5.0
ACTIVITY_LIMITS = RuntimeLimits(
    admission_timeout_s=ACTIVITY_TIMEOUT_S, idle_input_timeout_s=ACTIVITY_TIMEOUT_S
)


class GatedAdapter(InteractionAdapter):
    """Blocks on the first unit until released, emitting a fixed reply per unit."""

    def __init__(self, reply: list[OutputEvent]) -> None:
        self.reply = reply
        self.units: list[Unit] = []
        self.output_sink: OutputSink | None = None
        self.has_started = asyncio.Event()
        self.release = asyncio.Event()

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        self.output_sink = emit

    async def process(self, unit: Unit) -> int:
        assert self.output_sink is not None
        self.units.append(unit)
        for event in self.reply:
            await self.output_sink(event)
        self.has_started.set()
        await self.release.wait()
        return unit.real_samples

    async def close(self) -> None:
        pass


class SlowOpenAdapter(GatedAdapter):
    """Takes longer than the admission timeout to open its pipeline session."""

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        await asyncio.sleep(ACTIVITY_TIMEOUT_S * 2)
        await super().open(session_id, config, emit)


class SlowClearAdapter(GatedAdapter):
    """Let a clear request cross the previous idle deadline."""

    async def clear(self) -> int:
        await asyncio.sleep(ACTIVITY_TIMEOUT_S)
        return 0


class RejectedOpenAdapter(GatedAdapter):
    """Reject admission after the original deadline, with optional slow cleanup."""

    def __init__(self, cleanup_delay_s: float) -> None:
        super().__init__([])
        self.cleanup_delay_s = cleanup_delay_s

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        await asyncio.sleep(ACTIVITY_TIMEOUT_S * 2.5)
        raise RuntimeError("pipeline unavailable")

    async def close(self) -> None:
        await asyncio.sleep(self.cleanup_delay_s)


@asynccontextmanager
async def running_runtime(runtime: SessionRuntime) -> AsyncIterator[SessionRuntime]:
    runtime.start()
    try:
        yield runtime
    finally:
        await asyncio.wait_for(runtime.close("client_closed"), TEST_WAIT_TIMEOUT_S)


@asynccontextmanager
async def open_runtime(
    adapter: GatedAdapter, limits: RuntimeLimits | None = None
) -> AsyncIterator[SessionRuntime]:
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, limits or RuntimeLimits()
    )
    async with running_runtime(runtime):
        await runtime.update({}, "client_update")
        yield runtime


async def receive_until(
    runtime: SessionRuntime, event_type: type[Drained | UnitCompleted | Closed]
) -> list[Envelope]:
    envelopes: list[Envelope] = []
    deadline_s = time.monotonic() + TEST_WAIT_TIMEOUT_S
    async with aclosing(runtime.outputs()) as outputs:
        while True:
            envelope = await asyncio.wait_for(
                anext(outputs), deadline_s - time.monotonic()
            )
            envelopes.append(envelope)
            if isinstance(envelope.event, event_type):
                return envelopes
            else:
                pass


@pytest.mark.asyncio
async def test_eos_marks_only_the_last_unit_when_end_arrives_mid_backlog() -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_S)
        await runtime.append(b"\1" * UNIT_BYTES * 2, 1, None, "client_append_1")
        await runtime.end("client_end")

        adapter.release.set()
        drained = (await receive_until(runtime, Drained))[-1].event

        assert [unit.eos for unit in adapter.units] == [False, False, True]
        assert isinstance(drained, Drained)
        assert (drained.accepted_end_ms, drained.consumed_ms) == (60.0, 60.0)


@pytest.mark.asyncio
async def test_close_finishes_only_responses_the_client_has_seen() -> None:
    adapter = GatedAdapter([ResponseStarted("seen"), ResponseStarted("unseen")])
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_S)
        adapter.release.set()
        for envelope in await receive_until(runtime, UnitCompleted):
            if envelope.event == ResponseStarted("seen"):
                runtime.output_buffer.before_send(envelope)
                runtime.output_buffer.sent(envelope)
            else:
                pass

        await runtime.close("client_closed")
        closing = [envelope.event for envelope in await receive_until(runtime, Closed)]

        finished = [event for event in closing if isinstance(event, ResponseFinished)]
        assert [event.response_id for event in finished] == ["seen"]
        assert finished[0].status == "cancelled"


@pytest.mark.asyncio
async def test_context_exhaustion_closes_session() -> None:
    message = "context_exhausted: thinker context length 8192 tokens exhausted"

    class FailingAdapter(GatedAdapter):
        async def process(self, unit: Unit) -> int:
            raise RuntimeError(message)

    runtime = await open_runtime(FailingAdapter([]))
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "append")
    envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)
    failures = [entry.event for entry in envelopes if isinstance(entry.event, Failure)]
    assert len(failures) == 1
    assert failures[0].code == "context_exhausted"
    assert failures[0].is_fatal
    assert message in failures[0].message


def test_output_budget_counts_outbound_events_only() -> None:
    buffer = OutputBuffer(RuntimeLimits(max_output_bytes=1024, max_output_events=2))
    unit = Unit(0, 0, bytes(32000), 16000, images=(bytes(512 * 1024),) * 4)
    completed = Envelope(event=UnitCompleted(unit.unit_id), unit=unit)
    buffer.enqueue(completed)
    with pytest.raises(RuntimeError, match="outbound event budget exhausted"):
        buffer.enqueue(Envelope(event=AudioDelta("response", "item", bytes(2048))))
    buffer.enqueue(completed)
    with pytest.raises(RuntimeError, match="outbound event budget exhausted"):
        buffer.enqueue(completed)
    assert [buffer.dequeue(), buffer.dequeue(), buffer.dequeue()] == [
        completed,
        completed,
        None,
    ]


def closing_failure(envelopes: list[Envelope]) -> tuple[str, str]:
    failures = [
        envelope.event for envelope in envelopes if isinstance(envelope.event, Failure)
    ]
    closed = envelopes[-1].event
    assert len(failures) == 1 and failures[0].is_fatal and isinstance(closed, Closed)
    return failures[0].code, closed.reason


@pytest.mark.asyncio
async def test_slow_pipeline_open_is_not_an_admission_timeout() -> None:
    adapter = SlowOpenAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, ACTIVITY_LIMITS
    )
    async with running_runtime(runtime):
        await runtime.update({}, "client_update")

        assert runtime.state == "OPEN"


@pytest.mark.parametrize("cleanup_delay_s", [0, ACTIVITY_TIMEOUT_S * 1.25])
@pytest.mark.parametrize("should_retry", [False, True])
@pytest.mark.asyncio
async def test_failed_admission_grants_full_retry_window(
    cleanup_delay_s: float, should_retry: bool
) -> None:
    adapters = iter([RejectedOpenAdapter(cleanup_delay_s), GatedAdapter([])])
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: next(adapters), ACTIVITY_LIMITS
    )
    async with running_runtime(runtime):
        with pytest.raises(ProtocolError) as rejection:
            await runtime.update({}, "client_update")
        rejected_at_s = time.monotonic()
        assert rejection.value.code == "admission_rejected"

        await asyncio.sleep(ACTIVITY_TIMEOUT_S * 0.75)
        assert runtime.state == "CREATED"
        if should_retry:
            await runtime.update({}, "client_retry")
            assert runtime.state == "OPEN"
        else:
            envelopes = await receive_until(runtime, Closed)
            assert closing_failure(envelopes) == (
                "admission_timeout",
                "admission_timeout",
            )
            assert time.monotonic() - rejected_at_s >= ACTIVITY_TIMEOUT_S


@pytest.mark.asyncio
async def test_steady_input_keeps_session_open_past_idle_timeout() -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter, ACTIVITY_LIMITS) as runtime:
        # Note (Haiyang Luo): one-sample appends isolate activity from unit completion.
        for sequence in range(8):
            await runtime.append(b"\1\0", sequence, None, f"client_append_{sequence}")
            await asyncio.sleep(ACTIVITY_TIMEOUT_S / 2)

        assert runtime.state == "OPEN" and not adapter.units


@pytest.mark.asyncio
async def test_clear_refreshes_idle_deadline_before_adapter_returns() -> None:
    adapter = SlowClearAdapter([])
    async with open_runtime(
        adapter, RuntimeLimits(idle_input_timeout_s=ACTIVITY_TIMEOUT_S * 2)
    ) as runtime:
        await asyncio.sleep(ACTIVITY_TIMEOUT_S * 1.5)

        await runtime.clear("client_clear")
        await asyncio.sleep(0)

        assert runtime.state == "OPEN"


@pytest.mark.parametrize(
    "idle_timeout_s", [ACTIVITY_TIMEOUT_S / 2, ACTIVITY_TIMEOUT_S * 2]
)
@pytest.mark.asyncio
async def test_clear_completion_starts_full_idle_window(idle_timeout_s: float) -> None:
    async with open_runtime(
        SlowClearAdapter([]),
        RuntimeLimits(idle_input_timeout_s=idle_timeout_s),
    ) as runtime:
        await runtime.clear("client_clear")
        await asyncio.sleep(idle_timeout_s * 0.75)

        assert runtime.state == "OPEN"
        envelopes = await receive_until(runtime, Closed)
        assert closing_failure(envelopes) == ("idle_timeout", "idle_timeout")


@pytest.mark.asyncio
async def test_unit_in_flight_is_not_idle_and_idle_resumes_after_it() -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter, ACTIVITY_LIMITS) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_S)

        await asyncio.sleep(ACTIVITY_TIMEOUT_S * 3)
        assert runtime.state == "OPEN"

        adapter.release.set()
        envelopes = await receive_until(runtime, Closed)
        assert closing_failure(envelopes) == ("idle_timeout", "idle_timeout")


@pytest.mark.asyncio
async def test_unit_completing_as_idle_deadline_expires_keeps_session_open() -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter, ACTIVITY_LIMITS) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_S)

        adapter.release.set()
        # Note (Haiyang Luo): stall the loop to resume completion alongside an expired deadline.
        time.sleep(ACTIVITY_TIMEOUT_S * 1.5)
        await asyncio.sleep(ACTIVITY_TIMEOUT_S / 4)

        assert runtime.state == "OPEN"


@pytest.mark.asyncio
async def test_idle_timeout_shorter_than_admission_is_honored_after_open() -> None:
    limits = RuntimeLimits(
        admission_timeout_s=ACTIVITY_TIMEOUT_S * 4,
        idle_input_timeout_s=ACTIVITY_TIMEOUT_S,
    )
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: GatedAdapter([]), limits
    )
    async with running_runtime(runtime):
        # Note (Haiyang Luo): let the watchdog start waiting in CREATED before opening.
        await asyncio.sleep(ACTIVITY_TIMEOUT_S / 8)
        await runtime.update({}, "client_update")
        opened_s = time.monotonic()

        envelopes = await receive_until(runtime, Closed)

        assert closing_failure(envelopes) == ("idle_timeout", "idle_timeout")
        assert time.monotonic() - opened_s < ACTIVITY_TIMEOUT_S * 3
