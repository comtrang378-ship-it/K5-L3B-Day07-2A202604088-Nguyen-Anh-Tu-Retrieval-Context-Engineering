"""Append UTF-8 JSONL records for retrieval and agent runs."""

import json
import re
import threading
from pathlib import Path
from typing import Any

_WRITE_LOCK = threading.Lock()

_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?token|password|secret|authorization)"
    r"(\s*[:=]\s*)[^\s,;]+"
)
_BEARER_TOKEN = re.compile(r"(?i)\bBearer\s+\S+")
_API_TOKEN = re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b")
_URL_CREDENTIALS = re.compile(r"(?i)(://[^:/@\s]+:)[^@/\s]+(@)")


def sanitize_error_message(message: str) -> str:
    """Redact common credential forms before errors are written to local traces."""
    sanitized = _SECRET_ASSIGNMENT.sub(r"\1\2[REDACTED]", message)
    sanitized = _BEARER_TOKEN.sub("Bearer [REDACTED]", sanitized)
    sanitized = _URL_CREDENTIALS.sub(r"\1[REDACTED]\2", sanitized)
    return _API_TOKEN.sub("[REDACTED]", sanitized)


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append one complete JSON object; persistence errors are intentionally raised."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
    with _WRITE_LOCK:
        with destination.open("a", encoding="utf-8", newline="\n") as output:
            output.write(line)
            output.flush()
