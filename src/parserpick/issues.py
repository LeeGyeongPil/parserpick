"""A language-independent finding: a stable code plus parameters for the message table."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass
class Issue:
    """code is stable and never translated; `key` and `params` build the localized message."""

    code: str
    severity: str
    key: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def message_key(self) -> str:
        return self.key or f"issue.{self.code}"
