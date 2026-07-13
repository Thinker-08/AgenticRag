from __future__ import annotations

import asyncio
from typing import Awaitable, Callable

import structlog

from ..contracts import Job

log = structlog.get_logger("agrag.jobs")


class JobQueue:
    def __init__(
        self, handler: Callable[[Job, bytes], Awaitable[None]], concurrency: int = 2
    ) -> None:
        self._handler = handler
        self._q: asyncio.Queue[tuple[Job, bytes]] = asyncio.Queue()
        self._sem = asyncio.Semaphore(concurrency)
        self._tasks: set[asyncio.Task] = set()

    async def enqueue(self, job: Job, data: bytes) -> None:
        await self._q.put((job, data))
        task = asyncio.create_task(self._drain_one())
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _drain_one(self) -> None:
        job, data = await self._q.get()
        try:
            async with self._sem:
                await self._handler(job, data)
        except Exception:
            log.exception("job.handler_crashed", job_id=job.job_id, doc_id=job.doc_id)
        finally:
            self._q.task_done()

    async def join(self) -> None:
        await self._q.join()
