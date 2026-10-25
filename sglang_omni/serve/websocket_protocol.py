# SPDX-License-Identifier: Apache-2.0
"""Bound transport draining after a WebSocket application exits."""

import asyncio

from uvicorn.protocols.websockets.auto import AutoWebSocketsProtocol


class BoundedCloseWebSocketProtocol(AutoWebSocketsProtocol):
    """Abort sockets that outlive the server protocol's close timeout."""

    transport_abort_timer: asyncio.TimerHandle | None = None
    has_lost_connection: bool = False

    async def run_asgi(self) -> None:
        try:
            await super().run_asgi()
        finally:
            if self.has_lost_connection:
                pass
            else:
                self.transport_abort_timer = self.loop.call_later(
                    self.close_timeout, self.transport.abort
                )

    def connection_lost(self, exc: Exception | None) -> None:
        self.has_lost_connection = True
        if self.transport_abort_timer is not None:
            self.transport_abort_timer.cancel()
            self.transport_abort_timer = None
        else:
            pass
        super().connection_lost(exc)
