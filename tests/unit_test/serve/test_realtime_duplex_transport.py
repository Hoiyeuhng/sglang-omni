# SPDX-License-Identifier: Apache-2.0
"""Connection lifecycle tests for /v1/realtime over a real uvicorn socket."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

import pytest
import pytest_asyncio
import uvicorn
from websockets.asyncio.client import ClientConnection, connect

from sglang_omni.client.client import Client
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.manager import (
    RealtimeDeployment,
    RealtimeSessionManager,
)
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits
from tests.unit_test.serve.test_realtime_duplex_session import (
    MODEL_NAME,
    HealthCoordinator,
    ScriptedAdapter,
)

PING_INTERVAL_S = 0.2
PING_TIMEOUT_S = 0.2
RELEASE_MARGIN_S = 2.0
# note (Haiyang Luo): legacy websockets waits its default 10s close_timeout for the peer's close frame; uvicorn cannot set it.
LEGACY_CLOSE_TIMEOUT_S = 10.0
IDLE_PING_ROUNDS = 5
WS_CLOSE_TIMEOUT_S = {
    "websockets": LEGACY_CLOSE_TIMEOUT_S,
    "websockets-sansio": 0.0,
    "wsproto": 0.0,
}


@dataclass(kw_only=True)
class LiveServer:
    url: str
    adapter: ScriptedAdapter
    manager: RealtimeSessionManager
    release_deadline_s: float


async def start_live_server(
    ws_implementation: str,
) -> tuple[uvicorn.Server, LiveServer]:
    adapter = ScriptedAdapter()
    deployment = RealtimeDeployment(
        capabilities=Capabilities(),
        adapter_factory=lambda: adapter,
        limits=RuntimeLimits(),
        max_connections=1,
    )
    app = create_app(
        Client(HealthCoordinator(True)),
        model_name=MODEL_NAME,
        realtime_deployment=deployment,
    )
    # note (Haiyang Luo): mirrors the launcher config, with pings shortened so a dead peer surfaces fast.
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        timeout_keep_alive=120,
        ws=ws_implementation,
        ws_ping_interval=PING_INTERVAL_S,
        ws_ping_timeout=PING_TIMEOUT_S,
    )
    server = uvicorn.Server(config)
    serve_task = asyncio.create_task(server.serve())
    while not server.started:
        assert not serve_task.done(), "uvicorn exited during startup"
        await asyncio.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    return server, LiveServer(
        url=f"ws://127.0.0.1:{port}/v1/realtime",
        adapter=adapter,
        manager=app.state.realtime_manager,
        release_deadline_s=PING_INTERVAL_S
        + PING_TIMEOUT_S
        + WS_CLOSE_TIMEOUT_S[ws_implementation]
        + RELEASE_MARGIN_S,
    )


@pytest_asyncio.fixture(params=list(WS_CLOSE_TIMEOUT_S))
async def live_server(request: pytest.FixtureRequest) -> AsyncIterator[LiveServer]:
    if request.param == "wsproto":
        pytest.importorskip("wsproto")
    else:
        pass
    server, live = await start_live_server(request.param)
    try:
        yield live
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


async def wait_for_release(live: LiveServer) -> None:
    deadline_s = asyncio.get_running_loop().time() + live.release_deadline_s
    while live.manager.sessions or not live.adapter.is_closed:
        assert (
            asyncio.get_running_loop().time() < deadline_s
        ), f"session not released: sessions={list(live.manager.sessions)} adapter_closed={live.adapter.is_closed}"
        await asyncio.sleep(0.02)


async def assert_slot_reusable(live: LiveServer) -> None:
    async with connect(live.url, ping_interval=None) as connection:
        await receive_until(connection, "session.created")


@pytest.mark.asyncio
async def test_client_close_releases_session(live_server: LiveServer) -> None:
    async with connect(live_server.url, ping_interval=None) as connection:
        await open_session(connection)
        await connection.send(
            json.dumps({"event_id": "client_close", "type": "session.close"})
        )
        await receive_until(connection, "session.closed")
    await wait_for_release(live_server)
    await assert_slot_reusable(live_server)


@pytest.mark.asyncio
async def test_abrupt_socket_drop_releases_session(live_server: LiveServer) -> None:
    connection = await connect(live_server.url, ping_interval=None)
    await open_session(connection)
    connection.transport.abort()
    await wait_for_release(live_server)
    await assert_slot_reusable(live_server)


@pytest.mark.asyncio
async def test_unresponsive_peer_is_reaped_by_ping_timeout(
    live_server: LiveServer,
) -> None:
    connection = await connect(live_server.url, ping_interval=None)
    await open_session(connection)
    # A peer that stops reading never answers pings, like a client behind a dead NAT.
    connection.transport.pause_reading()
    try:
        await wait_for_release(live_server)
    finally:
        connection.transport.abort()
    await assert_slot_reusable(live_server)


@pytest.mark.asyncio
async def test_idle_live_peer_holds_session_without_timeout(
    live_server: LiveServer,
) -> None:
    """A peer that answers pings but sends nothing keeps its session and slot."""
    async with connect(live_server.url, ping_interval=None) as connection:
        await open_session(connection)
        await asyncio.sleep(IDLE_PING_ROUNDS * (PING_INTERVAL_S + PING_TIMEOUT_S))
        assert list(live_server.manager.sessions) and not live_server.adapter.is_closed
