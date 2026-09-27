import asyncio
import json
import queue
import threading

import websockets


class EventClient:
    def __init__(self, uri: str):
        self._uri = uri
        self._queue: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        asyncio.run(self._listen())

    async def _listen(self) -> None:
        while True:
            try:
                async with websockets.connect(self._uri) as websocket:
                    async for message in websocket:
                        event = json.loads(message)
                        self._queue.put(event)
            except Exception:
                await asyncio.sleep(2.0)

    def drain(self) -> list:
        events = []
        while True:
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events