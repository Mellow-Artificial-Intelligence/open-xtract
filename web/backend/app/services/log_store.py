"""In-memory log store for streaming extraction logs."""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator


@dataclass
class LogEntry:
    """A single log entry."""
    timestamp: str
    level: str
    message: str

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "level": self.level,
            "message": self.message,
        }


@dataclass
class ExtractionLogs:
    """Logs for a single extraction."""
    entries: list[LogEntry] = field(default_factory=list)
    completed: bool = False
    subscribers: list[asyncio.Queue] = field(default_factory=list)


class LogStore:
    """Thread-safe in-memory log store with pub/sub support."""

    def __init__(self):
        self._logs: dict[str, ExtractionLogs] = defaultdict(ExtractionLogs)
        self._lock = asyncio.Lock()

    async def add_log(
        self,
        extraction_id: str,
        message: str,
        level: str = "info",
    ) -> None:
        """Add a log entry and notify subscribers."""
        entry = LogEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            level=level,
            message=message,
        )

        async with self._lock:
            logs = self._logs[extraction_id]
            logs.entries.append(entry)

            for queue in logs.subscribers:
                await queue.put(entry)

    async def mark_completed(self, extraction_id: str) -> None:
        """Mark extraction as completed and notify subscribers."""
        async with self._lock:
            if extraction_id in self._logs:
                logs = self._logs[extraction_id]
                logs.completed = True
                for queue in logs.subscribers:
                    await queue.put(None)

    async def subscribe(self, extraction_id: str) -> AsyncGenerator[LogEntry | None, None]:
        """Subscribe to log updates for an extraction."""
        queue: asyncio.Queue = asyncio.Queue()

        async with self._lock:
            logs = self._logs[extraction_id]
            for entry in logs.entries:
                yield entry

            if logs.completed:
                return

            logs.subscribers.append(queue)

        try:
            while True:
                entry = await queue.get()
                if entry is None:
                    break
                yield entry
        finally:
            async with self._lock:
                if extraction_id in self._logs:
                    try:
                        self._logs[extraction_id].subscribers.remove(queue)
                    except ValueError:
                        pass

    async def get_logs(self, extraction_id: str) -> list[dict]:
        """Get all logs for an extraction."""
        async with self._lock:
            if extraction_id not in self._logs:
                return []
            return [e.to_dict() for e in self._logs[extraction_id].entries]

    async def cleanup(self, extraction_id: str) -> None:
        """Remove logs for an extraction (call after some time)."""
        async with self._lock:
            if extraction_id in self._logs:
                del self._logs[extraction_id]


log_store = LogStore()
