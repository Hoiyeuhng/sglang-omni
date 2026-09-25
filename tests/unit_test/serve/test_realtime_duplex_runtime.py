# SPDX-License-Identifier: Apache-2.0
"""Session runtime contracts that depend on scheduling the wire cannot pin."""

from __future__ import annotations

import asyncio
import time

import pytest

from sglang_omni.serve.realtime.control import Closed, Drained, Failure, UnitCompleted
from sglang_omni.serve.realtime.output import (
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
)
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.schema import SessionConfiguration
from sglang_omni.serve.realtime.types import (
    Capabilities,
    Envelope,
    InteractionAdapter,
    OutputSink,
    RuntimeLimits,
    Unit,
)

MODEL_NAME = "duplex-test"
SAMPLE_RATE = 16000
NATIVE_UNIT_MS = 20
UNIT_BYTES = SAMPLE_RATE * NATIVE_UNIT_MS // 1000 * 2
ACTIVITY_TIMEOUT_S = 0.2
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


async def open_runtime(
    adapter: GatedAdapter, limits: RuntimeLimits | None = None
) -> SessionRuntime:
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, limits or RuntimeLimits()
    )
    await runtime.update({}, "client_update")
    return runtime


async def receive_until(
    runtime: SessionRuntime, event_type: type[Drained | UnitCompleted | Closed]
) -> list[Envelope]:
    envelopes: list[Envelope] = []
    async for envelope in runtime.outputs():
        envelopes.append(envelope)
        if isinstance(envelope.event, event_type):
            break
        else:
            pass
    return envelopes


@pytest.mark.asyncio
async def test_eos_marks_only_the_last_unit_when_end_arrives_mid_backlog() -> None:
    adapter = GatedAdapter([])
    runtime = await open_runtime(adapter)
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
    await adapter.has_started.wait()
    await runtime.append(b"\1" * UNIT_BYTES * 2, 1, None, "client_append_1")
    await runtime.end("client_end")

    adapter.release.set()
    drained = (await receive_until(runtime, Drained))[-1].event
    await runtime.close("client_closed")

    assert [unit.eos for unit in adapter.units] == [False, False, True]
    assert isinstance(drained, Drained)
    assert (drained.accepted_end_ms, drained.consumed_ms) == (60.0, 60.0)


@pytest.mark.asyncio
async def test_close_finishes_only_responses_the_client_has_seen() -> None:
    adapter = GatedAdapter([ResponseStarted("seen"), ResponseStarted("unseen")])
    runtime = await open_runtime(adapter)
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
    await adapter.has_started.wait()
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


def closing_failure(envelopes: list[Envelope]) -> tuple[str, str]:
    failures = [
        envelope.event for envelope in envelopes if isinstance(envelope.event, Failure)
    ]
    closed = envelopes[-1].event
    assert len(failures) == 1 and failures[0].is_fatal and isinstance(closed, Closed)
    return failures[0].code, closed.reason


@pytest.mark.asyncio
async def test_session_without_update_is_closed_by_admission_timeout() -> None:
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: GatedAdapter([]), ACTIVITY_LIMITS
    )
    runtime.start()

    envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)

    assert closing_failure(envelopes) == ("admission_timeout", "admission_timeout")
    assert runtime.state == "CLOSED"


@pytest.mark.asyncio
async def test_slow_pipeline_open_is_not_an_admission_timeout() -> None:
    adapter = SlowOpenAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, ACTIVITY_LIMITS
    )
    runtime.start()

    await runtime.update({}, "client_update")

    assert runtime.state == "OPEN"
    await runtime.close("client_closed")


@pytest.mark.asyncio
async def test_open_session_without_input_is_closed_by_idle_timeout() -> None:
    adapter = GatedAdapter([])
    runtime = await open_runtime(adapter, ACTIVITY_LIMITS)
    runtime.start()

    envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)

    assert closing_failure(envelopes) == ("idle_timeout", "idle_timeout")
    assert runtime.state == "CLOSED"


@pytest.mark.asyncio
async def test_steady_input_keeps_session_open_past_idle_timeout() -> None:
    adapter = GatedAdapter([])
    runtime = await open_runtime(adapter, ACTIVITY_LIMITS)
    runtime.start()

    # One-sample appends never complete a unit, so only the appends themselves keep it open.
    for sequence in range(8):
        await runtime.append(b"\1\0", sequence, None, f"client_append_{sequence}")
        await asyncio.sleep(ACTIVITY_TIMEOUT_S / 2)

    assert runtime.state == "OPEN" and not adapter.units
    await runtime.close("client_closed")


@pytest.mark.asyncio
async def test_unit_in_flight_is_not_idle_and_idle_resumes_after_it() -> None:
    adapter = GatedAdapter([])
    runtime = await open_runtime(adapter, ACTIVITY_LIMITS)
    runtime.start()
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
    await adapter.has_started.wait()

    await asyncio.sleep(ACTIVITY_TIMEOUT_S * 3)
    assert runtime.state == "OPEN"

    adapter.release.set()
    envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)
    assert closing_failure(envelopes) == ("idle_timeout", "idle_timeout")


@pytest.mark.asyncio
async def test_unit_completing_as_idle_deadline_expires_keeps_session_open() -> None:
    adapter = GatedAdapter([])
    runtime = await open_runtime(adapter, ACTIVITY_LIMITS)
    runtime.start()
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
    await adapter.has_started.wait()

    adapter.release.set()
    # A blocked loop makes the unit completion and the expired deadline resume together.
    time.sleep(ACTIVITY_TIMEOUT_S * 1.5)
    await asyncio.sleep(ACTIVITY_TIMEOUT_S / 4)

    assert runtime.state == "OPEN"
    await runtime.close("client_closed")
