# SPDX-License-Identifier: Apache-2.0
"""Session runtime contracts that depend on scheduling the wire cannot pin."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import aclosing, asynccontextmanager
from pathlib import Path

import pytest

import sglang_omni.serve.realtime.runtime as runtime_module
from sglang_omni.profiler.duplex_events import SessionUnitReadyObservation
from sglang_omni.profiler.event_recorder import get_recorder
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
RUNTIME_LOGGER_NAME = "sglang_omni.serve.realtime.runtime"
ACTIVITY_TIMEOUT_SECONDS = 0.5
TEST_WAIT_TIMEOUT_SECONDS = 5.0
ACTIVITY_LIMITS = RuntimeLimits(
    session_update_timeout_s=ACTIVITY_TIMEOUT_SECONDS,
    idle_input_timeout_s=ACTIVITY_TIMEOUT_SECONDS,
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


class SlowAdmissionAdapter(GatedAdapter):
    """Reports when admission starts and holds it until released."""

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        self.has_started.set()
        await self.release.wait()
        await super().open(session_id, config, emit)


class SlowOpenAdapter(GatedAdapter):
    """The adapter opens the pipeline session after the admission timeout."""

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        await asyncio.sleep(ACTIVITY_TIMEOUT_SECONDS * 2)
        await super().open(session_id, config, emit)


class SlowClearAdapter(GatedAdapter):
    """The clear request completes after the previous idle deadline."""

    async def clear(self) -> int:
        await asyncio.sleep(ACTIVITY_TIMEOUT_SECONDS)
        return 0


class RejectedOpenAdapter(GatedAdapter):
    """The adapter rejects admission after the deadline and can delay cleanup."""

    def __init__(self, cleanup_delay_seconds: float) -> None:
        super().__init__([])
        self.cleanup_delay_seconds: float = cleanup_delay_seconds

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        await asyncio.sleep(ACTIVITY_TIMEOUT_SECONDS * 2.5)
        raise RuntimeError("pipeline unavailable")

    async def close(self) -> None:
        await asyncio.sleep(self.cleanup_delay_seconds)


@asynccontextmanager
async def running_runtime(runtime: SessionRuntime) -> AsyncIterator[SessionRuntime]:
    runtime.start()
    try:
        yield runtime
    finally:
        await asyncio.wait_for(
            runtime.close("client_closed"), TEST_WAIT_TIMEOUT_SECONDS
        )


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
    deadline_seconds = time.monotonic() + TEST_WAIT_TIMEOUT_SECONDS
    async with aclosing(runtime.outputs()) as outputs:
        while True:
            envelope = await asyncio.wait_for(
                anext(outputs), deadline_seconds - time.monotonic()
            )
            envelopes.append(envelope)
            if isinstance(envelope.event, event_type):
                return envelopes
            else:
                pass


def ready_observation(
    timestamp_ns: int, run_id: str = "run-a"
) -> SessionUnitReadyObservation:
    return SessionUnitReadyObservation(timestamp_ns=timestamp_ns, run_id=run_id)


@pytest.mark.asyncio
async def test_eos_marks_only_the_last_unit_when_end_arrives_mid_backlog() -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)
        await runtime.append(b"\1" * UNIT_BYTES * 2, 1, None, "client_append_1")
        await runtime.end("client_end")

        adapter.release.set()
        drained = (await receive_until(runtime, Drained))[-1].event

        assert [unit.eos for unit in adapter.units] == [False, False, True]
        assert isinstance(drained, Drained)
        assert (drained.accepted_end_ms, drained.consumed_ms) == (60.0, 60.0)


@pytest.mark.asyncio
async def test_ready_timestamp_survives_serial_pump_wait(monkeypatch) -> None:
    ready_observations = iter(
        (ready_observation(100), ready_observation(200), ready_observation(300))
    )
    monkeypatch.setattr(
        runtime_module,
        "capture_session_unit_ready",
        lambda: next(ready_observations),
    )
    adapter = GatedAdapter([])
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await adapter.has_started.wait()
        await runtime.append(b"\1" * UNIT_BYTES, 1, None, "client_append_1")
        await runtime.end("client_end")

        adapter.release.set()
        await receive_until(runtime, Drained)
        await runtime.close("client_closed")

        assert [unit.ready_observation.timestamp_ns for unit in adapter.units] == [
            100,
            200,
        ]


@pytest.mark.asyncio
async def test_ready_before_profiler_start_is_not_retrofitted(tmp_path: Path) -> None:
    recorder = get_recorder()
    if recorder.is_active():
        recorder.stop()
    else:
        pass
    adapter = GatedAdapter([])
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES * 2, 0, None, "client_append_0")
        await adapter.has_started.wait()

        recorder.start("run-a", str(tmp_path), "coordinator")
        await runtime.end("client_end")
        adapter.release.set()
        await receive_until(runtime, Drained)
        await runtime.close("client_closed")
        recorder.stop()

        assert [unit.ready_observation for unit in adapter.units] == [None, None]


@pytest.mark.asyncio
async def test_multiple_units_from_one_append_share_ready_timestamp(
    monkeypatch,
) -> None:
    ready_observations = iter((ready_observation(100), ready_observation(200)))
    monkeypatch.setattr(
        runtime_module,
        "capture_session_unit_ready",
        lambda: next(ready_observations),
    )
    adapter = GatedAdapter([])
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES * 2, 0, None, "client_append_0")
        await adapter.has_started.wait()
        await runtime.end("client_end")

        adapter.release.set()
        await receive_until(runtime, Drained)
        await runtime.close("client_closed")

        assert [unit.ready_observation.timestamp_ns for unit in adapter.units] == [
            100,
            100,
        ]


@pytest.mark.parametrize("has_empty_tail", [False, True])
@pytest.mark.asyncio
async def test_eos_marks_terminal_units_ready_at_eos(
    monkeypatch: pytest.MonkeyPatch, has_empty_tail: bool
) -> None:
    ready_observations = iter((ready_observation(100), ready_observation(200)))
    monkeypatch.setattr(
        runtime_module,
        "capture_session_unit_ready",
        lambda: next(ready_observations),
    )
    adapter = GatedAdapter([])
    async with open_runtime(adapter) as runtime:
        append_bytes = UNIT_BYTES if has_empty_tail else UNIT_BYTES + UNIT_BYTES // 2
        await runtime.append(b"\1" * append_bytes, 0, None, "client_append_0")
        await adapter.has_started.wait()
        await runtime.end("client_end")

        adapter.release.set()
        await receive_until(runtime, Drained)
        await runtime.close("client_closed")

        assert [unit.ready_observation.timestamp_ns for unit in adapter.units] == [
            100,
            200,
        ]
        if has_empty_tail:
            assert [unit.real_samples for unit in adapter.units] == [
                UNIT_BYTES // 2,
                0,
            ]
        else:
            pass


@pytest.mark.asyncio
async def test_optional_image_does_not_delay_audio_ready_timestamp(
    monkeypatch,
) -> None:
    ready_observations = iter(
        (ready_observation(100), ready_observation(200), ready_observation(300))
    )
    monkeypatch.setattr(
        runtime_module,
        "capture_session_unit_ready",
        lambda: next(ready_observations),
    )
    adapter = GatedAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(input_modalities=("audio", "image")),
        lambda: adapter,
        RuntimeLimits(),
    )
    await runtime.update({}, "client_update")
    await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
    await adapter.has_started.wait()
    await runtime.append(b"\1" * UNIT_BYTES, 1, None, "client_append_1")
    await runtime.append_image(b"\xff\xd8frame", 20, "client_image")
    await runtime.end("client_end")
    adapter.release.set()
    await receive_until(runtime, Drained)

    await runtime.close("client_closed")

    assert adapter.units[1].images == (b"\xff\xd8frame",)
    assert [unit.ready_observation.timestamp_ns for unit in adapter.units] == [100, 200]


@pytest.mark.asyncio
async def test_close_finishes_only_responses_the_client_has_seen() -> None:
    adapter = GatedAdapter([ResponseStarted("seen"), ResponseStarted("unseen")])
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)
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
@pytest.mark.parametrize(
    ("failure_message", "failure_code", "has_traceback"),
    [
        (
            "context_exhausted: the session reached the thinker context length",
            "context_exhausted",
            False,
        ),
        ("talker step failed", "internal", True),
    ],
)
async def test_unit_failure_closes_session_and_logs_once(
    failure_message: str,
    failure_code: str,
    has_traceback: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FailingAdapter(GatedAdapter):
        async def process(self, unit: Unit) -> int:
            raise RuntimeError(failure_message)

    with caplog.at_level(logging.WARNING, logger=RUNTIME_LOGGER_NAME):
        async with open_runtime(FailingAdapter([])) as runtime:
            await runtime.append(b"\1" * UNIT_BYTES, 0, None, "append")
            envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)
    failures = [entry.event for entry in envelopes if isinstance(entry.event, Failure)]
    assert len(failures) == 1
    assert failures[0].code == failure_code
    assert failures[0].is_fatal
    assert failure_message in failures[0].message
    runtime_records = [
        record for record in caplog.records if record.name == RUNTIME_LOGGER_NAME
    ]
    assert len(runtime_records) == 1
    assert runtime_records[0].levelno == logging.ERROR
    assert (runtime_records[0].exc_info is not None) == has_traceback


@pytest.mark.parametrize(
    "termination", ["cancelled_future", "cancelled_task", "client_close"]
)
@pytest.mark.asyncio
async def test_process_cancellation_releases_session_resources(
    termination: Literal["cancelled_future", "cancelled_task", "client_close"],
) -> None:
    class CancellableAdapter(GatedAdapter):
        def __init__(self) -> None:
            super().__init__([])
            self.close_call_count: int = 0

        async def process(self, unit: Unit) -> int:
            self.has_started.set()
            if termination == "cancelled_future":
                consumed_samples: asyncio.Future[int] = (
                    asyncio.get_running_loop().create_future()
                )
                consumed_samples.cancel()
                return await consumed_samples
            else:
                await self.release.wait()
                return unit.real_samples

        async def close(self) -> None:
            self.close_call_count += 1

    adapter = CancellableAdapter()
    async with open_runtime(adapter) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "append")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)
        if termination == "cancelled_task":
            assert runtime.pump_task is not None
            runtime.pump_task.cancel()
        elif termination == "client_close":
            await runtime.close("client_closed")
        else:
            pass

        envelopes = await receive_until(runtime, Closed)
        if termination == "client_close":
            assert envelopes[-1].event == Closed("client_closed")
            assert not any(
                isinstance(envelope.event, Failure) for envelope in envelopes
            )
        else:
            assert closing_failure(envelopes) == ("internal", "internal")
        assert adapter.close_call_count == 1
        assert runtime.state == "CLOSED"


@pytest.mark.parametrize("cancellation_stage", ["open", "admission_cleanup"])
@pytest.mark.parametrize("should_close_first", [False, True])
@pytest.mark.asyncio
async def test_cancelled_admission_closes_resources_and_rejects_queued_update(
    cancellation_stage: Literal["open", "admission_cleanup"],
    should_close_first: bool,
) -> None:
    class CancellableAdmissionAdapter(GatedAdapter):
        def __init__(self) -> None:
            super().__init__([])
            self.cleanup_started: asyncio.Event = asyncio.Event()
            self.close_call_count: int = 0
            self.is_resource_open: bool = False

        async def open(
            self, session_id: str, config: SessionConfiguration, emit: OutputSink
        ) -> None:
            self.is_resource_open = True
            self.has_started.set()
            if cancellation_stage == "admission_cleanup":
                raise RuntimeError("pipeline unavailable")
            else:
                await self.release.wait()

        async def close(self) -> None:
            self.close_call_count += 1
            if cancellation_stage == "admission_cleanup" and self.close_call_count == 1:
                self.cleanup_started.set()
                await self.release.wait()
            else:
                pass
            self.is_resource_open = False

    adapter = CancellableAdmissionAdapter()
    factory_call_count = 0

    def create_adapter() -> InteractionAdapter:
        nonlocal factory_call_count
        factory_call_count += 1
        return adapter

    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), create_adapter, RuntimeLimits()
    )
    async with running_runtime(runtime):
        update_task = asyncio.create_task(runtime.update({}, "open"))
        queued_update_task: asyncio.Task[None] | None = None
        client_close_task: asyncio.Task[None] | None = None
        try:
            cancellation_started = (
                adapter.cleanup_started
                if cancellation_stage == "admission_cleanup"
                else adapter.has_started
            )
            await asyncio.wait_for(
                cancellation_started.wait(), TEST_WAIT_TIMEOUT_SECONDS
            )
            queued_update_task = asyncio.create_task(runtime.update({}, "queued"))
            await asyncio.sleep(0)
            if should_close_first:
                client_close_task = asyncio.create_task(runtime.close("client_closed"))
                await asyncio.sleep(0)
            else:
                pass
            update_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(update_task, TEST_WAIT_TIMEOUT_SECONDS)
            with pytest.raises(ProtocolError, match="closing"):
                await asyncio.wait_for(queued_update_task, TEST_WAIT_TIMEOUT_SECONDS)

            envelopes = await receive_until(runtime, Closed)
            if should_close_first:
                assert envelopes[-1].event == Closed("client_closed")
                assert not any(
                    isinstance(envelope.event, Failure) for envelope in envelopes
                )
            else:
                assert closing_failure(envelopes) == ("internal", "internal")
            assert factory_call_count == 1
            assert adapter.close_call_count == (
                2 if cancellation_stage == "admission_cleanup" else 1
            )
            assert not adapter.is_resource_open
            assert runtime.state == "CLOSED"
        finally:
            update_task.cancel()
            pending_tasks = [update_task]
            if queued_update_task is not None:
                queued_update_task.cancel()
                pending_tasks.append(queued_update_task)
            else:
                pass
            if client_close_task is not None:
                pending_tasks.append(client_close_task)
            else:
                pass
            await asyncio.gather(*pending_tasks, return_exceptions=True)


@pytest.mark.parametrize("cleanup_failure", ["exception", "timeout"])
@pytest.mark.asyncio
async def test_failed_admission_cleanup_rejects_queued_update_and_releases_resources(
    cleanup_failure: Literal["exception", "timeout"],
) -> None:
    class FailedCleanupAdapter(GatedAdapter):
        def __init__(self) -> None:
            super().__init__([])
            self.cleanup_started: asyncio.Event = asyncio.Event()
            self.close_call_count: int = 0
            self.is_resource_open: bool = False

        async def open(
            self, session_id: str, config: SessionConfiguration, emit: OutputSink
        ) -> None:
            self.is_resource_open = True
            raise RuntimeError("pipeline unavailable")

        async def close(self) -> None:
            self.close_call_count += 1
            if self.close_call_count == 1:
                self.cleanup_started.set()
                await self.release.wait()
                raise RuntimeError("pipeline cleanup failed")
            else:
                self.is_resource_open = False

    adapter = FailedCleanupAdapter()
    factory_call_count = 0

    def create_adapter() -> InteractionAdapter:
        nonlocal factory_call_count
        factory_call_count += 1
        return adapter

    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(),
        create_adapter,
        RuntimeLimits(cleanup_timeout_s=ACTIVITY_TIMEOUT_SECONDS),
    )
    async with running_runtime(runtime):
        update_task = asyncio.create_task(runtime.update({}, "open"))
        queued_update_task: asyncio.Task[None] | None = None
        try:
            await asyncio.wait_for(
                adapter.cleanup_started.wait(), TEST_WAIT_TIMEOUT_SECONDS
            )
            queued_update_task = asyncio.create_task(runtime.update({}, "queued"))
            await asyncio.sleep(0)
            if cleanup_failure == "exception":
                adapter.release.set()
            else:
                pass

            with pytest.raises(RuntimeError, match="admission cleanup failed"):
                await asyncio.wait_for(update_task, TEST_WAIT_TIMEOUT_SECONDS)
            with pytest.raises(ProtocolError, match="closing"):
                await asyncio.wait_for(queued_update_task, TEST_WAIT_TIMEOUT_SECONDS)

            envelopes = await receive_until(runtime, Closed)
            assert closing_failure(envelopes) == ("internal", "internal")
            assert factory_call_count == 1
            assert adapter.close_call_count == 2
            assert not adapter.is_resource_open
            assert runtime.state == "CLOSED"
        finally:
            update_task.cancel()
            pending_tasks = [update_task]
            if queued_update_task is not None:
                queued_update_task.cancel()
                pending_tasks.append(queued_update_task)
            else:
                pass
            await asyncio.gather(*pending_tasks, return_exceptions=True)


@pytest.mark.asyncio
async def test_update_deadline_does_not_cut_off_an_admission_in_progress() -> None:
    adapter = SlowAdmissionAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(),
        lambda: adapter,
        ACTIVITY_LIMITS,
    )
    runtime.start()
    update_task = asyncio.create_task(runtime.update({}, "update"))
    await adapter.has_started.wait()
    await asyncio.sleep(ACTIVITY_TIMEOUT_SECONDS * 2)
    adapter.release.set()
    await asyncio.wait_for(update_task, 5)
    await asyncio.sleep(0)
    await runtime.close("client_closed")
    envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)

    assert [
        entry.event for entry in envelopes if isinstance(entry.event, Failure)
    ] == []
    assert envelopes[-1].event == Closed("client_closed")


@pytest.mark.asyncio
async def test_update_after_the_deadline_is_rejected_without_admission() -> None:
    adapter = SlowAdmissionAdapter([])
    adapter.release.set()
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(),
        lambda: adapter,
        ACTIVITY_LIMITS,
    )
    runtime.start()
    time.sleep(ACTIVITY_TIMEOUT_SECONDS * 1.25)

    await runtime.update({}, "update")
    envelopes = await asyncio.wait_for(receive_until(runtime, Closed), 5)

    assert not adapter.has_started.is_set()
    failures = [entry.event for entry in envelopes if isinstance(entry.event, Failure)]
    assert [failure.code for failure in failures] == ["session_update_timeout"]


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


@pytest.mark.asyncio
async def test_competing_closes_keep_initial_failure_reason() -> None:
    async with open_runtime(GatedAdapter([])) as runtime:
        runtime.fail("context full", "context_exhausted")
        await asyncio.wait_for(
            asyncio.gather(runtime.close("disconnect"), runtime.close("client_closed")),
            TEST_WAIT_TIMEOUT_SECONDS,
        )

    assert runtime.close_reason == "context_exhausted"


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


@pytest.mark.parametrize("cleanup_delay_seconds", [0, ACTIVITY_TIMEOUT_SECONDS * 1.25])
@pytest.mark.asyncio
async def test_failed_admission_after_deadline_closes_without_retry(
    cleanup_delay_seconds: float,
) -> None:
    adapter = RejectedOpenAdapter(cleanup_delay_seconds)
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: adapter, ACTIVITY_LIMITS
    )
    async with running_runtime(runtime):
        await runtime.update({}, "client_update")
        with pytest.raises(ProtocolError, match="closing"):
            await runtime.update({}, "client_retry")
        envelopes = await receive_until(runtime, Closed)
        assert closing_failure(envelopes) == (
            "session_update_timeout",
            "session_update_timeout",
        )


@pytest.mark.asyncio
async def test_repeated_failed_admissions_keep_original_deadline() -> None:
    class RejectedAdapter(GatedAdapter):
        async def open(
            self, session_id: str, config: SessionConfiguration, emit: OutputSink
        ) -> None:
            raise RuntimeError("pipeline unavailable")

    factory_call_count = 0

    def create_adapter() -> InteractionAdapter:
        nonlocal factory_call_count
        factory_call_count += 1
        return RejectedAdapter([])

    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), create_adapter, ACTIVITY_LIMITS
    )
    async with running_runtime(runtime):
        for attempt_index in range(3):
            with pytest.raises(ProtocolError) as rejection:
                await runtime.update({}, f"attempt_{attempt_index}")
            assert rejection.value.code == "admission_rejected"
            # Note (Haiyang Luo): Keep the watchdog asleep to exercise the next command's deadline fence.
            time.sleep(ACTIVITY_TIMEOUT_SECONDS * 0.4)
        await runtime.update({}, "late_retry")
        with pytest.raises(ProtocolError, match="closing"):
            await runtime.update({}, "after_close")

        envelopes = await receive_until(runtime, Closed)
        assert closing_failure(envelopes) == (
            "session_update_timeout",
            "session_update_timeout",
        )
        assert factory_call_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("is_valid_image", [False, True])
async def test_only_accepted_images_extend_idle_timeout(is_valid_image: bool) -> None:
    adapter = GatedAdapter([])
    runtime = SessionRuntime(
        MODEL_NAME,
        Capabilities(input_modalities=("audio", "image")),
        lambda: adapter,
        ACTIVITY_LIMITS,
    )
    async with running_runtime(runtime):
        await runtime.update({}, "client_update")
        await asyncio.sleep(ACTIVITY_TIMEOUT_SECONDS / 2)
        if is_valid_image:
            await runtime.append_image(b"\xff\xd8frame", 0, "client_image")
        else:
            with pytest.raises(ProtocolError, match="image must be JPEG or PNG"):
                await runtime.append_image(b"invalid", 0, "client_image")

        image_sent_seconds = time.monotonic()
        envelopes = await receive_until(runtime, Closed)
        assert closing_failure(envelopes) == ("idle_timeout", "idle_timeout")
        idle_after_image_seconds = time.monotonic() - image_sent_seconds
        if is_valid_image:
            assert idle_after_image_seconds >= ACTIVITY_TIMEOUT_SECONDS
        else:
            assert idle_after_image_seconds < ACTIVITY_TIMEOUT_SECONDS


@pytest.mark.asyncio
async def test_steady_input_keeps_session_open_past_idle_timeout() -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter, ACTIVITY_LIMITS) as runtime:
        # Note (Haiyang Luo): One sample updates input activity without a complete input unit.
        for sequence in range(8):
            await runtime.append(b"\1\0", sequence, None, f"client_append_{sequence}")
            await asyncio.sleep(ACTIVITY_TIMEOUT_SECONDS / 2)

        assert runtime.state == "OPEN" and not adapter.units


@pytest.mark.asyncio
async def test_clear_refreshes_idle_deadline_before_adapter_returns() -> None:
    adapter = SlowClearAdapter([])
    async with open_runtime(
        adapter, RuntimeLimits(idle_input_timeout_s=ACTIVITY_TIMEOUT_SECONDS * 2)
    ) as runtime:
        await asyncio.sleep(ACTIVITY_TIMEOUT_SECONDS * 1.5)

        await runtime.clear("client_clear")
        await asyncio.sleep(0)

        assert runtime.state == "OPEN"


@pytest.mark.parametrize(
    "idle_timeout_seconds", [ACTIVITY_TIMEOUT_SECONDS / 2, ACTIVITY_TIMEOUT_SECONDS * 2]
)
@pytest.mark.asyncio
async def test_clear_completion_starts_full_idle_window(
    idle_timeout_seconds: float,
) -> None:
    async with open_runtime(
        SlowClearAdapter([]),
        RuntimeLimits(idle_input_timeout_s=idle_timeout_seconds),
    ) as runtime:
        await runtime.clear("client_clear")
        await asyncio.sleep(idle_timeout_seconds * 0.75)

        assert runtime.state == "OPEN"
        envelopes = await receive_until(runtime, Closed)
        assert closing_failure(envelopes) == ("idle_timeout", "idle_timeout")


@pytest.mark.asyncio
async def test_unit_in_flight_is_not_idle_and_idle_resumes_after_it() -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter, ACTIVITY_LIMITS) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)

        await asyncio.sleep(ACTIVITY_TIMEOUT_SECONDS * 3)
        assert runtime.state == "OPEN"

        adapter.release.set()
        envelopes = await receive_until(runtime, Closed)
        assert closing_failure(envelopes) == ("idle_timeout", "idle_timeout")


@pytest.mark.asyncio
async def test_unit_completing_as_idle_deadline_expires_keeps_session_open() -> None:
    adapter = GatedAdapter([])
    async with open_runtime(adapter, ACTIVITY_LIMITS) as runtime:
        await runtime.append(b"\1" * UNIT_BYTES, 0, None, "client_append_0")
        await asyncio.wait_for(adapter.has_started.wait(), TEST_WAIT_TIMEOUT_SECONDS)

        adapter.release.set()
        # Note (Haiyang Luo): This pause puts unit completion after the idle deadline.
        time.sleep(ACTIVITY_TIMEOUT_SECONDS * 1.5)
        await asyncio.sleep(ACTIVITY_TIMEOUT_SECONDS / 4)

        assert runtime.state == "OPEN"


@pytest.mark.asyncio
async def test_idle_timeout_shorter_than_admission_is_honored_after_open() -> None:
    limits = RuntimeLimits(
        session_update_timeout_s=ACTIVITY_TIMEOUT_SECONDS * 4,
        idle_input_timeout_s=ACTIVITY_TIMEOUT_SECONDS,
    )
    runtime = SessionRuntime(
        MODEL_NAME, Capabilities(), lambda: GatedAdapter([]), limits
    )
    async with running_runtime(runtime):
        # Note (Haiyang Luo): The watchdog must start its wait in CREATED before the session opens.
        await asyncio.sleep(ACTIVITY_TIMEOUT_SECONDS / 8)
        await runtime.update({}, "client_update")
        opened_seconds = time.monotonic()

        envelopes = await receive_until(runtime, Closed)

        assert closing_failure(envelopes) == ("idle_timeout", "idle_timeout")
        assert time.monotonic() - opened_seconds < ACTIVITY_TIMEOUT_SECONDS * 3
