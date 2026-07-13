"""In-process conversation store (C16) — the local stand-in for Redis/Postgres session state."""

from __future__ import annotations

from ...contracts import Conversation, Turn


class MemorySessionStore:
    def __init__(self) -> None:
        self._c: dict[tuple[str, str], Conversation] = {}

    async def get(self, tenant_id: str, session_id: str) -> Conversation:
        return self._c.get((tenant_id, session_id)) or Conversation(
            session_id=session_id, tenant_id=tenant_id
        )

    async def append(
        self, tenant_id: str, session_id: str, turn: Turn, *, max_turns: int = 20
    ) -> None:
        convo = await self.get(tenant_id, session_id)
        turns = (convo.turns + [turn])[-max_turns:]
        self._c[(tenant_id, session_id)] = convo.model_copy(update={"turns": turns})
