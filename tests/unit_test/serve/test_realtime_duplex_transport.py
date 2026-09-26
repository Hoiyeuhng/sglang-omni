# SPDX-License-Identifier: Apache-2.0
"""Connection lifecycle tests for /v1/realtime over a real uvicorn socket."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version

import pytest
import pytest_asyncio
import uvicorn
from packaging.version import Version

# The asyncio client API was added in websockets 13; the project floor is 12.
pytest.importorskip("websockets", minversion="13.0")

from websockets.asyncio.client import ClientConnection, connect  # noqa: E402
from websockets.exceptions import ConnectionClosedError  # noqa: E402

from sglang_omni.client.client import Client  # noqa: E402
from sglang_omni.serve.openai_api import create_app  # noqa: E402
from sglang_omni.serve.realtime.manager import (  # noqa: E402
    RealtimeDeployment,
    RealtimeSessionManager,
)
from sglang_omni.serve.realtime.schema import JsonObject  # noqa: E402
from sglang_omni.serve.realtime.types import Capabilities, RuntimeLimits  # noqa: E402
from tests.unit_test.serve.test_realtime_duplex_session import (  # noqa: E402
    MODEL_NAME,
    HealthCoordinator,
    ScriptedAdapter,
)

PING_INTERVAL_S = 0.2
PING_TIMEOUT_S = 0.2
RELEASE_MARGIN_S = 2.0
# note (Haiyang Luo): legacy websockets waits its default 10s close_timeout for the peer's close frame; uvicorn cannot set it.
LEGACY_CLOSE_TIMEOUT_S = 10.0
IDLE_PING_ROUNDS = 3
IDLE_INPUT_TIMEOUT_S = IDLE_PING_ROUNDS * (PING_INTERVAL_S + PING_TIMEOUT_S)
WS_CLOSE_TIMEOUT_S = {
    "websockets": LEGACY_CLOSE_TIMEOUT_S,
    "websockets-sansio": 0.0,
    "wsproto": 0.0,
}
# Server keepalive pings arrived in uvicorn 0.44 for sansio and 0.46 for wsproto; the project floor is 0.23.
MIN_UVICORN_FOR_PINGS = {
    "websockets": Version("0.23.0"),
    "websockets-sansio": Version("0.44.0"),
    "wsproto": Version("0.46.0"),
}


@dataclass(kw_only=True)
class LiveServer:
    url: str
    adapter: ScriptedAdapter
    manager: RealtimeSessionManager
    release_deadline_s: float


@asynccontextmanager
async def serve_live(
    ws_implementation: str, limits: RuntimeLimits
) -> AsyncIterator[LiveServer]:
    uvicorn_version = Version(version("uvicorn"))
    if uvicorn_version < MIN_UVICORN_FOR_PINGS[ws_implementation]:
        pytest.skip(f"uvicorn {uvicorn_version} does not ping over {ws_implementation}")
    elif ws_implementation == "wsproto":
        pytest.importorskip("wsproto")
    else:
        pass
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
    try:
        yield LiveServer(
            url=f"ws://127.0.0.1:{port}/v1/realtime",
            adapter=adapter,
            manager=app.state.realtime_manager,
            release_deadline_s=PING_INTERVAL_S
            + PING_TIMEOUT_S
            + WS_CLOSE_TIMEOUT_S[ws_implementation]
            + RELEASE_MARGIN_S,
        )
    finally:
        server.should_exit = True
        await server.shutdown()


@pytest_asyncio.fixture(params=list(WS_CLOSE_TIMEOUT_S))
async def live_server(request: pytest.FixtureRequest) -> AsyncIterator[LiveServer]:
    async with serve_live(request.param, RuntimeLimits()) as live:
        yield live


@pytest_asyncio.fixture(params=list(WS_CLOSE_TIMEOUT_S))
async def idle_limited_server(
    request: pytest.FixtureRequest,
) -> AsyncIterator[LiveServer]:
    limits = RuntimeLimits(idle_input_timeout_s=IDLE_INPUT_TIMEOUT_S)
    async with serve_live(request.param, limits) as live:
        yield live


async def receive_until(connection: ClientConnection, event_type: str) -> None:
    while json.loads(await connection.recv())["type"] != event_type:
        pass


async def receive_all(connection: ClientConnection) -> list[JsonObject]:
    events: list[JsonObject] = []
    try:
        async for message in connection:
            events.append(json.loads(message))
    except ConnectionClosedError:
        # A non-1000 close code ends iteration with an error instead of a clean stop.
        pass
    return events


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
async def test_idle_live_peer_is_reaped_by_idle_timeout(
    idle_limited_server: LiveServer,
) -> None:
    """A peer that keeps answering pings but sends nothing is closed by the runtime."""
    async with connect(idle_limited_server.url, ping_interval=None) as connection:
        await open_session(connection)
        opened_s = time.monotonic()
        events = await asyncio.wait_for(
            receive_all(connection), IDLE_INPUT_TIMEOUT_S + RELEASE_MARGIN_S
        )
        closed_after_s = time.monotonic() - opened_s
        close_code, close_reason = connection.close_code, connection.close_reason

    error_codes = [
        event["error"]["code"] for event in events if event["type"] == "error"
    ]
    assert error_codes == ["idle_timeout"]
    assert events[-1] == {
        **events[-1],
        "type": "session.closed",
        "reason": "idle_timeout",
    }
    # Answering pings for several rounds must not count as client input.
    assert (
        IDLE_INPUT_TIMEOUT_S <= closed_after_s < IDLE_INPUT_TIMEOUT_S + RELEASE_MARGIN_S
    )
    assert (close_code, close_reason) == (1008, "idle_timeout")
    await wait_for_release(idle_limited_server)
    await assert_slot_reusable(idle_limited_server)
