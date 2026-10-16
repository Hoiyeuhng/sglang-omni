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
from sglang_omni.pipeline.coordinator import Coordinator
from sglang_omni.proto.session import SessionLimits
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
