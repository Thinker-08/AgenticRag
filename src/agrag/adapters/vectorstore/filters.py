from __future__ import annotations

from typing import Any

_OPS = {"$in", "$nin", "$gte", "$lte", "$gt", "$lt", "$ne"}


def field(chunk_meta: dict, key: str) -> Any:
    return chunk_meta.get(key)


def matches(chunk_meta: dict, filters: dict | None) -> bool:
    if not filters:
        return True

    for key, cond in filters.items():
        val = field(chunk_meta, key)
        if isinstance(cond, dict):
            for op, target in cond.items():
                if op not in _OPS:
                    continue
                if op == "$in" and val not in target:
                    return False
                if op == "$nin" and val in target:
                    return False
                if op == "$ne" and val == target:
                    return False
                if op in ("$gte", "$lte", "$gt", "$lt"):
                    if val is None:
                        return False
                    if op == "$gte" and not val >= target:
                        return False
                    if op == "$lte" and not val <= target:
                        return False
                    if op == "$gt" and not val > target:
                        return False
                    if op == "$lt" and not val < target:
                        return False
        else:
            if val != cond:
                return False

    return True
