import asyncio
import collections
import json
import logging

import websockets

logger = logging.getLogger("orchestrator.event_bus")


class EventBus:
    def __init__(self, history_size: int = 50):
        self._clients: set = set()
        self._history: collections.deque = collections.deque(maxlen=history_size)

    async def publish(self, event: dict) -> None:
        self._history.append(event)
        message = json.dumps(event)

        stale = set()
        for client in self._clients:
            try:
                await client.send(message)
            except websockets.exceptions.ConnectionClosed:
                stale.add(client)

        self._clients -= stale

    async def _handler(self, websocket) -> None:
        self._clients.add(websocket)
        logger.info("dashboard client connected")

        try:
            for event in self._history:
                await websocket.send(json.dumps(event))

            async for _ in websocket:
                pass
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info("dashboard client disconnected")

    async def serve(self, host: str, port: int) -> None:
        async with websockets.serve(self._handler, host, port):
            logger.info(f"event bus listening on ws://{host}:{port}")
            await asyncio.Future()