# SPDX-License-Identifier: Apache-2.0
"""Tests of session closure through a WebSocket connection."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
import uvicorn

from sglang_omni.client.client import Client
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.manager import (
    RealtimeDeployment,
    RealtimeSessionManager,
)
from sglang_omni.serve.realtime.output import AudioDelta, ResponseStarted
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits
from sglang_omni.serve.websocket_protocol import BoundedCloseWebSocketProtocol
from tests.unit_test.fixtures.realtime_fakes import (
    MODEL_NAME,
    HealthCoordinator,
    ScriptedAdapter,
)

# Note (Haiyang Luo): The asyncio client API is available in websockets 13 and later.
pytest.importorskip("websockets", minversion="13.0")

from websockets.asyncio.client import (  # noqa: E402 - websockets version gate
    ClientConnection,
    connect,
)

ADMISSION_TIMEOUT_SECONDS = 1.2
RELEASE_MARGIN_SECONDS = 2.0
SOCKET_WAIT_TIMEOUT_SECONDS = 5.0
STARTUP_POLL_INTERVAL_SECONDS = 0.01
RELEASE_POLL_INTERVAL_SECONDS = 0.02
TRANSPORT_CLOSE_TIMEOUT_SECONDS = 0.05
BACKPRESSURE_AUDIO_CHUNK_BYTES = 128 * 1024
BACKPRESSURE_AUDIO_CHUNKS = 128


@dataclass(kw_only=True)
class LiveServer:
    url: str
    adapter: ScriptedAdapter
    session_manager: RealtimeSessionManager
    server: uvicorn.Server


@asynccontextmanager
async def serve_live(
    limits: RuntimeLimits, capabilities: Capabilities | None = None
) -> AsyncIterator[LiveServer]:
    adapter = ScriptedAdapter()
    deployment = RealtimeDeployment(
        capabilities=capabilities or Capabilities(),
        adapter_factory=lambda: adapter,
        limits=limits,
        max_connections=1,
    )
    app = create_app(
        Client(HealthCoordinator(True)),
        model_name=MODEL_NAME,
        realtime_deployment=deployment,
    )
    config = uvicorn.Config(
        app,
        ws=BoundedCloseWebSocketProtocol,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        ws_ping_interval=None,
        ws_per_message_deflate=False,
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    startup_deadline_seconds = time.monotonic() + SOCKET_WAIT_TIMEOUT_SECONDS
    try:
        while not server.started:
            assert not serve_task.done(), "uvicorn exited during startup"
            assert (
                time.monotonic() < startup_deadline_seconds
            ), "uvicorn startup timed out"
            await asyncio.sleep(STARTUP_POLL_INTERVAL_SECONDS)
        port = server.servers[0].sockets[0].getsockname()[1]
        yield LiveServer(
            url=f"ws://127.0.0.1:{port}/v1/realtime",
            adapter=adapter,
            session_manager=app.state.realtime_manager,
            server=server,
        )
    finally:
        server.should_exit = True
        await asyncio.wait_for(serve_task, SOCKET_WAIT_TIMEOUT_SECONDS)


async def receive_until(connection: ClientConnection, event_type: str) -> None:
    deadline_seconds = time.monotonic() + SOCKET_WAIT_TIMEOUT_SECONDS
    while True:
        message = await asyncio.wait_for(
            connection.recv(), deadline_seconds - time.monotonic()
        )
        if json.loads(message)["type"] == event_type:
            return
        else:
            pass


async def assert_released(live_server: LiveServer, timeout_seconds: float) -> None:
    deadline_seconds = time.monotonic() + timeout_seconds
    while (
        live_server.session_manager.sessions
        or live_server.server.server_state.connections
    ):
        assert time.monotonic() < deadline_seconds, "connection was not released"
        await asyncio.sleep(RELEASE_POLL_INTERVAL_SECONDS)
    async with connect(live_server.url, ping_interval=None) as connection:
        await receive_until(connection, "session.created")


@pytest.mark.asyncio
async def test_client_close_releases_session() -> None:
    """Client-requested closure must include client_closed in the WebSocket close frame."""
    async with serve_live(RuntimeLimits()) as live_server:
        async with connect(live_server.url, ping_interval=None) as connection:
            await receive_until(connection, "session.created")
            await connection.send(
                json.dumps(
                    {
                        "event_id": "client_update",
                        "type": "session.update",
                        "session": {},
                    }
                )
            )
            await receive_until(connection, "session.updated")
            await connection.send(
                json.dumps({"event_id": "client_close", "type": "session.close"})
            )
            await receive_until(connection, "session.closed")
        assert (connection.close_code, connection.close_reason) == (
            1000,
            "client_closed",
        )
        await assert_released(live_server, RELEASE_MARGIN_SECONDS)
        assert live_server.adapter.is_closed


@pytest.mark.asyncio
async def test_admission_timeout_reports_policy_close_over_socket() -> None:
    """An initial session.update timeout must close the socket with 1008 policy violation."""
    limits = RuntimeLimits(session_update_timeout_s=ADMISSION_TIMEOUT_SECONDS)
    async with serve_live(limits) as live_server:
        async with connect(live_server.url, ping_interval=None) as connection:
            await receive_until(connection, "session.created")
            await asyncio.wait_for(
                connection.wait_closed(),
                ADMISSION_TIMEOUT_SECONDS + RELEASE_MARGIN_SECONDS,
            )

        assert (connection.close_code, connection.close_reason) == (
            1008,
            "session_update_timeout",
        )
        await assert_released(live_server, RELEASE_MARGIN_SECONDS)
        assert not live_server.adapter.is_closed


@pytest.mark.asyncio
async def test_nonreading_peer_releases_transport_after_session_close() -> None:
    """Closing a backpressured session must also release its TCP connection."""
    limits = RuntimeLimits(
        cleanup_timeout_s=TRANSPORT_CLOSE_TIMEOUT_SECONDS,
        max_output_bytes=64 * 1024 * 1024,
    )
    async with serve_live(
        limits, Capabilities(output_modalities=("audio",))
    ) as live_server:
        connection = await connect(
            live_server.url,
            max_queue=1,
            max_size=None,
            compression=None,
            ping_interval=None,
            close_timeout=TRANSPORT_CLOSE_TIMEOUT_SECONDS,
        )
        try:
            await receive_until(connection, "session.created")
            await connection.send(
                json.dumps(
                    {"event_id": "open", "type": "session.update", "session": {}}
                )
            )
            await receive_until(connection, "session.updated")
            protocol = next(iter(live_server.server.server_state.connections))
            protocol.close_timeout = TRANSPORT_CLOSE_TIMEOUT_SECONDS
            assert live_server.adapter.output_sink is not None
            await live_server.adapter.output_sink(ResponseStarted("response"))
            for _ in range(BACKPRESSURE_AUDIO_CHUNKS):
                await live_server.adapter.output_sink(
                    AudioDelta(
                        "response",
                        "item",
                        bytes(BACKPRESSURE_AUDIO_CHUNK_BYTES),
                    )
                )
            deadline_seconds = time.monotonic() + SOCKET_WAIT_TIMEOUT_SECONDS
            _, high_water_bytes = protocol.transport.get_write_buffer_limits()
            while protocol.transport.get_write_buffer_size() <= high_water_bytes:
                assert time.monotonic() < deadline_seconds, "send did not stall"
                await asyncio.sleep(RELEASE_POLL_INTERVAL_SECONDS)
            await connection.send(
                json.dumps({"event_id": "close", "type": "session.close"})
            )
            await assert_released(live_server, RELEASE_MARGIN_SECONDS)
            assert live_server.adapter.is_closed
        finally:
            connection.transport.abort()
            await connection.wait_closed()
