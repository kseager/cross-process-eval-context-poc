"""Dataset loading.

The dataset is a JSONL file where each line is an object with at least a
``query`` and a ``ground_truth`` field. An optional ``id`` correlates the row to
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

    ``ground_truth`` is kept as the raw parsed value (a dict/object, string,
    list, etc.) so the full ground-truth object can be attached to a span.
    """

    id: str
    query: str
    ground_truth: Any


def load_dataset(path: str | Path) -> Iterator[DatasetItem]:
    """Yield :class:`DatasetItem` objects from a JSONL file.

    Args:
        path: Path to a ``.jsonl`` dataset file.

    Yields:
        One :class:`DatasetItem` per non-empty line.

    Raises:
        FileNotFoundError: If the dataset file does not exist.
        ValueError: If a line is missing the required ``query`` or
            ``ground_truth`` fields.
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
            if "query" not in record or "ground_truth" not in record:
                raise ValueError(
                    f"Line {line_number} must contain 'query' and "
                    f"'ground_truth' fields."
                )

            yield DatasetItem(
                id=str(record.get("id", line_number)),
                query=str(record["query"]),
                ground_truth=record["ground_truth"],
            )
