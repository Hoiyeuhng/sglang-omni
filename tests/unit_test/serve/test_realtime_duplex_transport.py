# SPDX-License-Identifier: Apache-2.0
"""Connection lifecycle tests for /v1/realtime over a real uvicorn socket."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
import uvicorn

# The asyncio client API was added in websockets 13; the project floor is 12.
pytest.importorskip("websockets", minversion="13.0")

from websockets.asyncio.client import ClientConnection, connect  # noqa: E402

from sglang_omni.client.client import Client  # noqa: E402
from sglang_omni.serve.openai_api import create_app  # noqa: E402
from sglang_omni.serve.realtime.manager import (  # noqa: E402
    RealtimeDeployment,
    RealtimeSessionManager,
)
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits  # noqa: E402
from tests.unit_test.serve.test_realtime_duplex_session import (  # noqa: E402
    MODEL_NAME,
    HealthCoordinator,
    ScriptedAdapter,
)

PING_INTERVAL_S = 0.2
PING_TIMEOUT_S = 0.2
ACTIVITY_TIMEOUT_S = 3 * (PING_INTERVAL_S + PING_TIMEOUT_S)
RELEASE_MARGIN_S = 2.0
# Note (Haiyang Luo): uvicorn before 0.50 serves auto with legacy websockets, which waits
# its fixed 10s close_timeout for a dead peer's close frame.
DEAD_PEER_RELEASE_S = PING_INTERVAL_S + PING_TIMEOUT_S + 10 + RELEASE_MARGIN_S


@dataclass(kw_only=True)
class LiveServer:
    url: str
    adapter: ScriptedAdapter
    manager: RealtimeSessionManager


@asynccontextmanager
async def serve_live(limits: RuntimeLimits) -> AsyncIterator[LiveServer]:
    adapter = ScriptedAdapter()
    deployment = RealtimeDeployment(
        capabilities=Capabilities(),
        adapter_factory=lambda: adapter,
        limits=limits,
        max_connections=1,
    )
    app = create_app(
        Client(HealthCoordinator(True)),
        model_name=MODEL_NAME,
        realtime_deployment=deployment,
    )
    # Note (Haiyang Luo): the launcher config, with pings shortened so a dead peer surfaces fast.
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        timeout_keep_alive=120,
        ws_ping_interval=PING_INTERVAL_S,
        ws_ping_timeout=PING_TIMEOUT_S,
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        assert not serve_task.done(), "uvicorn exited during startup"
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        yield LiveServer(
            url=f"ws://127.0.0.1:{port}/v1/realtime",
            adapter=adapter,
            manager=app.state.realtime_manager,
        )
    finally:
        server.should_exit = True
        await server.shutdown()


async def receive_until(connection: ClientConnection, event_type: str) -> None:
    while json.loads(await connection.recv())["type"] != event_type:
        pass


async def open_session(connection: ClientConnection) -> None:
    await receive_until(connection, "session.created")
    await connection.send(
        json.dumps(
            {"event_id": "client_update", "type": "session.update", "session": {}}
        )
    )
    await receive_until(connection, "session.updated")


async def assert_released(live: LiveServer, deadline_s: float) -> None:
    give_up_s = time.monotonic() + deadline_s
    while live.manager.sessions:
        assert time.monotonic() < give_up_s, "session was not released"
        await asyncio.sleep(0.02)
    async with connect(live.url, ping_interval=None) as connection:
        await receive_until(connection, "session.created")


@pytest.mark.asyncio
async def test_client_close_releases_session() -> None:
    async with serve_live(RuntimeLimits()) as live:
        async with connect(live.url, ping_interval=None) as connection:
            await open_session(connection)
            await connection.send(
                json.dumps({"event_id": "client_close", "type": "session.close"})
            )
            await receive_until(connection, "session.closed")
        await assert_released(live, RELEASE_MARGIN_S)
        assert live.adapter.is_closed


@pytest.mark.asyncio
async def test_aborted_socket_releases_session() -> None:
    async with serve_live(RuntimeLimits()) as live:
        connection = await connect(live.url, ping_interval=None)
        await open_session(connection)
        connection.transport.abort()
        await assert_released(live, RELEASE_MARGIN_S)
        assert live.adapter.is_closed


@pytest.mark.asyncio
async def test_unresponsive_peer_is_reaped_by_ping_timeout() -> None:
    async with serve_live(RuntimeLimits()) as live:
        connection = await connect(live.url, ping_interval=None)
        await open_session(connection)
        # A peer that stops reading never answers pings, like a client behind a dead NAT.
        connection.transport.pause_reading()
        try:
            await assert_released(live, DEAD_PEER_RELEASE_S)
        finally:
            connection.transport.abort()
        assert live.adapter.is_closed


@pytest.mark.parametrize(
    ("is_updating", "timeout_code"),
    [(False, "admission_timeout"), (True, "idle_timeout")],
)
@pytest.mark.asyncio
async def test_silent_peer_answering_pings_is_closed_by_activity_timeout(
    is_updating: bool, timeout_code: str
) -> None:
    limits = RuntimeLimits(
        admission_timeout_s=ACTIVITY_TIMEOUT_S, idle_input_timeout_s=ACTIVITY_TIMEOUT_S
    )
    async with serve_live(limits) as live:
        connected_at_s = time.monotonic()
        async with connect(live.url, ping_interval=None) as connection:
            if is_updating:
                await open_session(connection)
            else:
                pass
            await asyncio.wait_for(
                connection.wait_closed(), ACTIVITY_TIMEOUT_S + RELEASE_MARGIN_S
            )
            closed_after_s = time.monotonic() - connected_at_s

        # Answering pings for several rounds must not count as client activity.
        assert closed_after_s >= ACTIVITY_TIMEOUT_S
        assert (connection.close_code, connection.close_reason) == (1008, timeout_code)
        await assert_released(live, RELEASE_MARGIN_S)
        assert live.adapter.is_closed is is_updating
