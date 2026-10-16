# SPDX-License-Identifier: Apache-2.0
"""Realtime lifecycle contracts through the coordinator session implementation."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Literal
from unittest.mock import AsyncMock

import pytest
from starlette.websockets import WebSocket

import sglang_omni.serve.realtime.adapters as adapter_module
from sglang_omni.client.client import Client
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto.session import SessionLimits
from sglang_omni.serve.realtime.control import Closed, Failure
from sglang_omni.serve.realtime.manager import (
    RealtimeDeployment,
    RealtimeSessionManager,
)
from sglang_omni.serve.realtime.output import TurnFailure
from sglang_omni.serve.realtime.protocol import SharedRealtimeSession
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits
from tests.unit_test.serve.test_realtime_coordinator_adapter import (
    PROCESS_TIMEOUT_S,
    SESSION_CONFIG,
    STAGES,
    RecordingSink,
    build_adapter,
    build_request,
)

IDLE_INPUT_TIMEOUT_SECONDS = 0.4
TRANSPORT_TIMEOUT_SECONDS = 0.2
TRANSPORT_FINISH_TIMEOUT_SECONDS = 1.0
ACTIVITY_COUNT = 8


@pytest.fixture
def pipeline_coordinator(monkeypatch: pytest.MonkeyPatch) -> Coordinator:
    coordinator = Coordinator("inproc://completion", "inproc://abort", STAGES[0])
    coordinator.register_stage(STAGES[0], "inproc://thinker")
    coordinator.running = True
    # Note (Haiyang Luo): Only stage execution is replaced; session timers and cleanup run normally.
    monkeypatch.setattr(coordinator, "session_operation", AsyncMock())
    return coordinator


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("pipeline_idle_timeout_seconds", "activity"),
    [(0.1, None), (0.4, "audio"), (0.8, "audio"), (0.1, "image")],
)
async def test_runtime_owns_pipeline_input_idle_timeout(
    pipeline_coordinator: Coordinator,
    pipeline_idle_timeout_seconds: float,
    activity: Literal["audio", "image"] | None,
) -> None:
    adapter = build_adapter(
        pipeline_coordinator,
        limits=SessionLimits(idle_timeout_s=pipeline_idle_timeout_seconds),
    )
    runtime = SessionRuntime(
        "test",
        Capabilities(
            input_modalities=("audio", "image"),
            image_frames_per_unit=(ACTIVITY_COUNT,),
        ),
        lambda: adapter,
        RuntimeLimits(idle_input_timeout_s=IDLE_INPUT_TIMEOUT_SECONDS),
    )
    websocket = AsyncMock(spec=WebSocket)
    runtime.start()
    sender = asyncio.create_task(SharedRealtimeSession(websocket, runtime).send())
    try:
        await runtime.update({}, "open")
        last_activity_seconds = time.monotonic()
        if activity is None:
            await asyncio.sleep(pipeline_idle_timeout_seconds * 2)
            assert runtime.state == "OPEN"
        else:
            interval_seconds = (
                max(pipeline_idle_timeout_seconds, IDLE_INPUT_TIMEOUT_SECONDS)
                * 1.5
                / ACTIVITY_COUNT
            )
            for sequence in range(ACTIVITY_COUNT):
                await asyncio.sleep(interval_seconds)
                assert runtime.state == "OPEN"
                if activity == "audio":
                    await runtime.append(b"\0\0", sequence, None, "audio")
                else:
                    await runtime.append_image(b"\xff\xd8", 0, "image")
                last_activity_seconds = time.monotonic()
        await asyncio.wait_for(sender, PROCESS_TIMEOUT_S)
        assert time.monotonic() - last_activity_seconds >= IDLE_INPUT_TIMEOUT_SECONDS
        websocket.close.assert_awaited_once_with(1008, "idle_timeout")
        server_events = [
            json.loads(call.args[0]) for call in websocket.send_text.await_args_list
        ]
        assert [
            event["error"]["code"]
            for event in server_events
            if event["type"] == "error"
        ] == ["idle_timeout"]
        assert server_events[-1]["type"] == "session.closed"
        assert not pipeline_coordinator.sessions
        reopened = await pipeline_coordinator.open_session(
            build_request(SESSION_CONFIG),
            stages=STAGES,
            session_id=runtime.session_id,
        )
        await pipeline_coordinator.close_session(reopened)
    finally:
        await runtime.close("disconnect")
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
        await pipeline_coordinator.stop_sessions()


@pytest.mark.asyncio
async def test_pipeline_backup_expires_despite_partial_input_activity(
    pipeline_coordinator: Coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(adapter_module, "DEFAULT_IDLE_TIMEOUT_S", 0.1)
    adapter = build_adapter(
        pipeline_coordinator,
        limits=SessionLimits(idle_timeout_s=0.1, operation_timeout_s=0.05),
    )
    idle_timeout_seconds = IDLE_INPUT_TIMEOUT_SECONDS / 2
    runtime = SessionRuntime(
        "test",
        Capabilities(),
        lambda: adapter,
        RuntimeLimits(
            idle_input_timeout_s=idle_timeout_seconds, cleanup_timeout_s=0.05
        ),
    )
    websocket = AsyncMock(spec=WebSocket)
    runtime.start()
    sender = asyncio.create_task(SharedRealtimeSession(websocket, runtime).send())
    last_input_seconds = time.monotonic()
    try:
        await runtime.update({}, "open")
        for sequence in range(ACTIVITY_COUNT):
            if runtime.state != "OPEN":
                break
            else:
                pass
            await runtime.append(b"\0\0", sequence, None, "audio")
            last_input_seconds = time.monotonic()
            await asyncio.sleep(idle_timeout_seconds / 3)
        await asyncio.wait_for(sender, PROCESS_TIMEOUT_S)

        assert time.monotonic() - last_input_seconds < idle_timeout_seconds
        websocket.close.assert_awaited_once_with(1011, "internal")
        assert not pipeline_coordinator.sessions
    finally:
        await runtime.close("disconnect")
        sender.cancel()
        await asyncio.gather(sender, return_exceptions=True)
        await pipeline_coordinator.stop_sessions()


@pytest.mark.asyncio
async def test_unmanaged_adapter_keeps_pipeline_idle_timeout(
    pipeline_coordinator: Coordinator,
) -> None:
    adapter = build_adapter(
        pipeline_coordinator, limits=SessionLimits(idle_timeout_s=0.1)
    )
    sink = RecordingSink()
    try:
        await adapter.open("unmanaged", SESSION_CONFIG, sink)
        await asyncio.wait_for(sink.has_published.wait(), PROCESS_TIMEOUT_S)
        assert isinstance(sink.published[0][0], TurnFailure)
        assert not pipeline_coordinator.sessions
    finally:
        await adapter.close()
        await pipeline_coordinator.stop_sessions()


@pytest.mark.asyncio
async def test_competing_closes_release_managed_session_once_and_keep_first_failure(
    pipeline_coordinator: Coordinator, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter = build_adapter(pipeline_coordinator)
    close_adapter = adapter.close
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def gated_close() -> None:
        close_started.set()
        await release_close.wait()
        await close_adapter()

    adapter_close = AsyncMock(side_effect=gated_close)
    monkeypatch.setattr(adapter, "close", adapter_close)
    runtime = SessionRuntime("test", Capabilities(), lambda: adapter, RuntimeLimits())
    runtime.start()
    competing_closes: list[asyncio.Task[None]] = []
    try:
        await runtime.update({}, "open")
        runtime.fail("context full", "context_exhausted")
        await asyncio.wait_for(close_started.wait(), PROCESS_TIMEOUT_S)
        competing_closes = [
            asyncio.create_task(runtime.close(reason))
            for reason in ("client_closed", "disconnect")
        ]
        await asyncio.sleep(0)
        runtime.fail("late failure", "internal")
        release_close.set()
        await asyncio.wait_for(
            asyncio.gather(runtime.close("disconnect"), *competing_closes),
            PROCESS_TIMEOUT_S,
        )
        runtime.fail("failure after close", "internal")
        assert [envelope.event async for envelope in runtime.outputs()] == [
            Failure("context_exhausted", "context full", True),
            Closed("context_exhausted"),
        ]
        assert runtime.close_reason == "context_exhausted"
        adapter_close.assert_awaited_once()
        assert not pipeline_coordinator.sessions
        assert (
            sum(
                operation.args[1] == "close"
                for operation in pipeline_coordinator.session_operation.await_args_list
            )
            == 1
        )
    finally:
        release_close.set()
        await asyncio.wait_for(
            asyncio.gather(runtime.close("disconnect"), *competing_closes),
            PROCESS_TIMEOUT_S,
        )
        await pipeline_coordinator.stop_sessions()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "termination", ["disconnect", "cancel", "write_failure", "worker_failure"]
)
async def test_connection_termination_releases_managed_pipeline_session(
    pipeline_coordinator: Coordinator,
    termination: Literal["disconnect", "cancel", "write_failure", "worker_failure"],
) -> None:
    adapter = build_adapter(pipeline_coordinator)
    runtime = SessionRuntime("test", Capabilities(), lambda: adapter, RuntimeLimits())
    incoming: asyncio.Queue[dict[str, str]] = asyncio.Queue()
    opened = asyncio.Event()

    async def send_text(message: str) -> None:
        if json.loads(message)["type"] == "session.updated":
            opened.set()
        else:
            pass

    websocket = AsyncMock(spec=WebSocket)
    websocket.receive.side_effect = incoming.get
    websocket.send_text.side_effect = send_text
    incoming.put_nowait(
        {
            "type": "websocket.receive",
            "text": json.dumps(
                {"event_id": "open", "type": "session.update", "session": {}}
            ),
        }
    )
    connection = asyncio.create_task(SharedRealtimeSession(websocket, runtime).run())
    try:
        await asyncio.wait_for(opened.wait(), PROCESS_TIMEOUT_S)
        assert runtime.session_id in pipeline_coordinator.sessions
        if termination == "cancel":
            connection.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(connection, PROCESS_TIMEOUT_S)
        elif termination == "write_failure":
            websocket.send_text.side_effect = RuntimeError("socket write failed")
            await runtime.append(b"\0\0", 0, None, "audio")
            with pytest.raises(RuntimeError, match="socket write failed"):
                await asyncio.wait_for(connection, PROCESS_TIMEOUT_S)
        else:
            if termination == "disconnect":
                incoming.put_nowait({"type": "websocket.disconnect"})
            else:
                await pipeline_coordinator.fail_sessions("worker exited")
            await asyncio.wait_for(connection, PROCESS_TIMEOUT_S)
        assert runtime.state == "CLOSED"
        assert not pipeline_coordinator.sessions
        if termination == "worker_failure":
            websocket.close.assert_awaited_once_with(1011, "internal")
            with pytest.raises(RuntimeError, match="Coordinator is not running"):
                await pipeline_coordinator.open_session(
                    build_request(SESSION_CONFIG), stages=STAGES
                )
        else:
            reopened = await pipeline_coordinator.open_session(
                build_request(SESSION_CONFIG),
                stages=STAGES,
                session_id=runtime.session_id,
            )
            await pipeline_coordinator.close_session(reopened)
    finally:
        connection.cancel()
        await asyncio.gather(connection, return_exceptions=True)
        await pipeline_coordinator.stop_sessions()


@pytest.mark.asyncio
async def test_outer_cancellation_survives_transport_cleanup_failure(
    pipeline_coordinator: Coordinator,
) -> None:
    adapter = build_adapter(pipeline_coordinator)
    runtime = SessionRuntime(
        "test",
        Capabilities(),
        lambda: adapter,
        RuntimeLimits(transport_timeout_s=TRANSPORT_TIMEOUT_SECONDS),
    )
    incoming: asyncio.Queue[dict[str, str]] = asyncio.Queue()
    send_started = asyncio.Event()
    send_cancelled = asyncio.Event()
    release_send = asyncio.Event()
    send_cancellation_count = 0

    async def send_text(message: str) -> None:
        nonlocal send_cancellation_count
        if json.loads(message)["type"] == "session.updated":
            send_started.set()
            while not release_send.is_set():
                try:
                    await release_send.wait()
                except asyncio.CancelledError:
                    send_cancellation_count += 1
                    send_cancelled.set()
        else:
            pass

    websocket = AsyncMock(spec=WebSocket)
    websocket.receive.side_effect = incoming.get
    websocket.send_text.side_effect = send_text
    session = SharedRealtimeSession(websocket, runtime)
    incoming.put_nowait(
        {
            "type": "websocket.receive",
            "text": json.dumps(
                {"event_id": "open", "type": "session.update", "session": {}}
            ),
        }
    )
    connection = asyncio.create_task(session.run())
    try:
        await asyncio.wait_for(send_started.wait(), PROCESS_TIMEOUT_S)
        connection.cancel("outer cancellation")
        await asyncio.wait_for(send_cancelled.wait(), PROCESS_TIMEOUT_S)
        connection.cancel("later cancellation")
        finished, _ = await asyncio.wait(
            (connection,), timeout=TRANSPORT_FINISH_TIMEOUT_SECONDS
        )
        assert connection in finished
        with pytest.raises(asyncio.CancelledError, match="outer cancellation") as error:
            await connection

        assert connection.cancelled()
        assert isinstance(error.value.__cause__, RuntimeError)
        assert str(error.value.__cause__) == (
            "local realtime task did not acknowledge cancellation"
        )
        assert send_cancellation_count == 2
        assert runtime.state == "CLOSED"
        assert not pipeline_coordinator.sessions
    finally:
        release_send.set()
        connection.cancel()
        tasks_to_finish = [connection]
        if session.send_task is not None:
            tasks_to_finish.append(session.send_task)
        else:
            pass
        await asyncio.wait_for(
            asyncio.gather(*tasks_to_finish, return_exceptions=True),
            PROCESS_TIMEOUT_S,
        )
        await pipeline_coordinator.stop_sessions()


@pytest.mark.parametrize(
    ("stalled_operation", "termination"),
    [
        ("send", "idle"),
        ("close", "idle"),
        ("close_ignores_first_cancel", "idle"),
        ("send", "adapter_failure"),
        ("close", "adapter_failure"),
        ("close_ignores_first_cancel", "adapter_failure"),
        ("send", "write_timeout"),
        ("close_ignores_first_cancel", "cancel_cleanup"),
    ],
)
@pytest.mark.asyncio
async def test_stalled_transport_releases_managed_session_and_connection_slot(
    pipeline_coordinator: Coordinator,
    monkeypatch: pytest.MonkeyPatch,
    stalled_operation: Literal["send", "close", "close_ignores_first_cancel"],
    termination: Literal["idle", "adapter_failure", "write_timeout", "cancel_cleanup"],
) -> None:
    adapter = build_adapter(pipeline_coordinator)
    capabilities = Capabilities()
    deployment = RealtimeDeployment(
        capabilities=capabilities,
        adapter_factory=lambda: adapter,
        limits=RuntimeLimits(
            cleanup_timeout_s=PROCESS_TIMEOUT_S,
            transport_timeout_s=TRANSPORT_TIMEOUT_SECONDS,
            idle_input_timeout_s=(
                PROCESS_TIMEOUT_S
                if termination == "write_timeout"
                else TRANSPORT_TIMEOUT_SECONDS / 2
            ),
        ),
        max_connections=1,
    )
    session_manager = RealtimeSessionManager(
        deployment=deployment,
        client=Client(pipeline_coordinator),
        model_name="test",
    )
    incoming: asyncio.Queue[dict[str, str]] = asyncio.Queue()
    opened = asyncio.Event()
    transport_stalled = asyncio.Event()
    release_transport = asyncio.Event()
    first_close_cancelled = asyncio.Event()

    async def send_text(message: str) -> None:
        event_type = json.loads(message)["type"]
        if event_type == "session.updated":
            opened.set()
        elif (
            stalled_operation == "send" and event_type == "sglang.input_audio.accepted"
        ):
            transport_stalled.set()
            await release_transport.wait()
        else:
            pass

    async def close(code: int = 1000, reason: str | None = None) -> None:
        if stalled_operation == "send":
            pass
        else:
            transport_stalled.set()
            try:
                await release_transport.wait()
            except asyncio.CancelledError:
                if stalled_operation == "close_ignores_first_cancel":
                    first_close_cancelled.set()
                    await release_transport.wait()
                else:
                    raise

    websocket = AsyncMock(spec=WebSocket)
    websocket.receive.side_effect = incoming.get
    websocket.send_text.side_effect = send_text
    websocket.close.side_effect = close
    session = session_manager.open(websocket)
    assert isinstance(session, SharedRealtimeSession)
    incoming.put_nowait(
        {
            "type": "websocket.receive",
            "text": json.dumps(
                {"event_id": "open", "type": "session.update", "session": {}}
            ),
        }
    )
    existing_tasks = asyncio.all_tasks()

    async def run_connection() -> None:
        try:
            await session.run()
        finally:
            await session_manager.close(session.session_id)

    connection = asyncio.create_task(run_connection())
    try:
        await asyncio.wait_for(opened.wait(), PROCESS_TIMEOUT_S)
        assert session.session_id in pipeline_coordinator.sessions
        if stalled_operation == "send":
            await session.runtime.append(b"\0\0", 0, None, "audio")
            await asyncio.wait_for(transport_stalled.wait(), PROCESS_TIMEOUT_S)
        else:
            pass
        if termination == "adapter_failure":
            monkeypatch.setattr(
                pipeline_coordinator,
                "session_operation",
                AsyncMock(side_effect=[RuntimeError("pipeline failed"), None]),
            )
            accepted_bytes = 2 if stalled_operation == "send" else 0
            append_sequence = 1 if stalled_operation == "send" else 0
            await session.runtime.append(
                bytes(capabilities.native_unit_bytes - accepted_bytes),
                append_sequence,
                None,
                "failed_audio",
            )
        else:
            pass
        await asyncio.wait_for(transport_stalled.wait(), PROCESS_TIMEOUT_S)
        if termination == "cancel_cleanup":
            await asyncio.wait_for(first_close_cancelled.wait(), PROCESS_TIMEOUT_S)
            connection.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(
                    asyncio.shield(connection), TRANSPORT_FINISH_TIMEOUT_SECONDS
                )
        else:
            await asyncio.wait_for(
                asyncio.shield(connection), TRANSPORT_FINISH_TIMEOUT_SECONDS
            )

        assert session.runtime.state == "CLOSED"
        if termination == "write_timeout":
            assert session.runtime.close_reason == "transport_timeout"
        elif termination == "adapter_failure":
            assert session.runtime.close_reason == "internal"
        else:
            assert session.runtime.close_reason == "idle_timeout"
        assert not session_manager.active_sessions()
        assert not pipeline_coordinator.sessions
        assert not (asyncio.all_tasks() - existing_tasks)
    finally:
        release_transport.set()
        connection.cancel()
        await asyncio.gather(connection, return_exceptions=True)
        await pipeline_coordinator.stop_sessions()
        remaining_tasks = asyncio.all_tasks() - existing_tasks
        for task in remaining_tasks:
            task.cancel()
        await asyncio.gather(*remaining_tasks, return_exceptions=True)
