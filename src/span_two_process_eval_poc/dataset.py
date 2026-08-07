"""Dataset loading.

The dataset is a JSONL file where each line is an object with a ``messages``
array (the standard agent-input contract: a list of ``{"role", "content"}``
turns) and a ``ground_truth`` field. An optional ``id`` correlates the row to
its evaluation; if missing, the line number is used.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class DatasetItem:
    """A single evaluation example.

    ``messages`` is the standard agent input: a list of ``{"role", "content"}``
    turns (single- or multi-turn). ``ground_truth`` is kept as the raw parsed
    value (a dict/object, string, list, etc.) so the full ground-truth object
    can be attached to a span.
    """

    id: str
    messages: list[dict[str, Any]]
    ground_truth: Any

    @property
    def user_text(self) -> str:
        """The text of the last user turn (for display / logging)."""
        for message in reversed(self.messages):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""


def _validate_messages(messages: Any, line_number: int) -> list[dict[str, Any]]:
    """Ensure ``messages`` is a non-empty list of ``{role, content}`` objects."""
    if not isinstance(messages, list) or not messages:
        raise ValueError(
            f"Line {line_number}: 'messages' must be a non-empty list of "
            f"{{'role', 'content'}} objects."
        )
    for message in messages:
        if (
            not isinstance(message, dict)
            or "role" not in message
            or "content" not in message
        ):
            raise ValueError(
                f"Line {line_number}: each message must be an object with "
                f"'role' and 'content' fields."
            )
    return messages


def load_dataset(path: str | Path) -> Iterator[DatasetItem]:
    """Yield :class:`DatasetItem` objects from a JSONL file.

    Args:
        path: Path to a ``.jsonl`` dataset file.

    Yields:
        One :class:`DatasetItem` per non-empty line.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If a line is missing the required ``messages`` or
            ``ground_truth`` fields, or ``messages`` is malformed.
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    with dataset_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            if "messages" not in record or "ground_truth" not in record:
                raise ValueError(
                    f"Line {line_number} must contain 'messages' and "
                    f"'ground_truth' fields."
                )

            yield DatasetItem(
                id=str(record.get("id", line_number)),
                messages=_validate_messages(record["messages"], line_number),
                ground_truth=record["ground_truth"],
            )
