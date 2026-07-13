from __future__ import annotations

from pydantic import BaseModel, Field

from .evidence import Citation


class Turn(BaseModel):
    role: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    carried_entities: list[str] = Field(default_factory=list)


class Conversation(BaseModel):
    session_id: str
    tenant_id: str
    turns: list[Turn] = Field(default_factory=list)

    def window(self, n: int = 6) -> list[Turn]:
        return self.turns[-n:]

    def last_assistant(self) -> Turn | None:
        for t in reversed(self.turns):
            if t.role == "assistant":
                return t
        return None
