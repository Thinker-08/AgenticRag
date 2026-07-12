from .output_filter import scan_answer
from .pii import detect_pii, scrub, scrub_attrs
from .sanitize import datamark, neutralize_template_tokens, strip_datamarks


class SecurityError(RuntimeError):
    """A fail-closed security boundary was violated (e.g. a query with no tenant scope, C31)."""


__all__ = [
    "SecurityError",
    "datamark",
    "detect_pii",
    "neutralize_template_tokens",
    "scan_answer",
    "scrub",
    "scrub_attrs",
    "strip_datamarks",
]
