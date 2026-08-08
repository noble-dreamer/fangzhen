"""Formal response-ID selection shared by inversion and output validation.

The training corpus uses weak-basis probe ordinals.  Formal inversion batches
must instead be selected from the numeric IDs embedded in the source response
filenames under ``frequency_response``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


FORMAL_RESPONSE_PATTERN = re.compile(
    r"^dataset_a_frequency_sample_(?P<sample_id>\d+)_H_complex\.npz$"
)


@dataclass(frozen=True)
class FormalSelection:
    """An ordered formal sample selection and its reproducible provenance."""

    sample_ids: tuple[int, ...]
    kind: str
    start_id: int | None = None
    end_id: int | None = None

    def __post_init__(self) -> None:
        if not self.sample_ids or any(sample_id <= 0 for sample_id in self.sample_ids):
            raise ValueError("Formal selection must contain positive sample IDs")
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise ValueError("Formal selection contains duplicate sample IDs")
        if self.kind not in {"configured", "explicit", "source_range"}:
            raise ValueError(f"Unsupported formal selection kind: {self.kind}")
        has_range = self.start_id is not None or self.end_id is not None
        if self.kind == "source_range":
            if self.start_id is None or self.end_id is None:
                raise ValueError("source_range selection requires both start_id and end_id")
            if self.start_id <= 0 or self.end_id < self.start_id:
                raise ValueError("Invalid inclusive formal sample-ID range")
            expected = tuple(range(self.start_id, self.end_id + 1))
            if self.sample_ids != expected:
                raise ValueError("source_range selection must contain every requested ID in order")
        elif has_range:
            raise ValueError("Only source_range selections may have start/end IDs")

    @property
    def is_source_range(self) -> bool:
        return self.kind == "source_range"

    @property
    def batch_directory_name(self) -> str:
        if not self.is_source_range or self.start_id is None or self.end_id is None:
            raise ValueError("Only source-ID range selections have a default batch directory")
        return f"ids_{self.start_id:06d}_{self.end_id:06d}"

    def to_contract(self) -> dict[str, object]:
        result: dict[str, object] = {
            "kind": self.kind,
            "sample_ids": list(self.sample_ids),
        }
        if self.is_source_range:
            result.update(
                {
                    "start_id": self.start_id,
                    "end_id": self.end_id,
                    "range_is_inclusive": True,
                    "source": "dataset_frequency_response_filename_ids",
                }
            )
        return result


def sample_name(sample_id: int) -> str:
    return f"dataset_a_frequency_sample_{sample_id:04d}"


def parse_sample_ids(values: Iterable[str | int] | None) -> tuple[int, ...]:
    """Parse the legacy explicit ``--sample-ids`` form, including ``1-4`` ranges."""

    if values is None:
        return ()
    result: list[int] = []
    for value in values:
        for token in str(value).split(","):
            token = token.strip()
            if not token:
                continue
            if "-" in token:
                first_text, last_text = token.split("-", maxsplit=1)
                first, last = int(first_text), int(last_text)
                if first <= 0 or last < first:
                    raise ValueError(f"Invalid sample range: {token}")
                result.extend(range(first, last + 1))
            else:
                sample_id = int(token)
                if sample_id <= 0:
                    raise ValueError(f"Sample IDs must be positive: {token}")
                result.append(sample_id)
    result = list(dict.fromkeys(result))
    if not result:
        raise ValueError("No sample IDs selected")
    return tuple(result)


def discover_formal_response_ids(dataset_path: Path) -> tuple[int, ...]:
    """Return numeric sample IDs that actually have a formal response file."""

    response_dir = dataset_path / "frequency_response"
    if not response_dir.is_dir():
        raise FileNotFoundError(f"Formal response directory does not exist: {response_dir}")
    found: dict[int, Path] = {}
    for path in response_dir.iterdir():
        if not path.is_file():
            continue
        match = FORMAL_RESPONSE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        sample_id = int(match.group("sample_id"))
        if sample_id <= 0:
            raise ValueError(f"Formal response ID must be positive: {path.name}")
        previous = found.get(sample_id)
        if previous is not None:
            raise RuntimeError(
                "Ambiguous formal response IDs after numeric parsing: "
                f"{previous.name} and {path.name} both map to {sample_id}"
            )
        found[sample_id] = path
    if not found:
        raise FileNotFoundError(f"No formal response files found under {response_dir}")
    return tuple(sorted(found))


def select_formal_sample_ids(
    *,
    dataset_path: Path,
    configured_ids: Iterable[int],
    explicit_ids: Iterable[str | int] | None = None,
    start_id: int | None = None,
    end_id: int | None = None,
) -> FormalSelection:
    """Select formal IDs without confusing them with training-basis probe IDs.

    ``start_id`` and ``end_id`` are an inclusive range of source response IDs.
    The range must be complete so a typo or an unfinished server export cannot
    silently produce a partial inversion batch.
    """

    if explicit_ids is not None and (start_id is not None or end_id is not None):
        raise ValueError("--sample-ids cannot be combined with --start-id or --end-id")
    if (start_id is None) != (end_id is None):
        raise ValueError("--start-id and --end-id must be supplied together")
    if start_id is not None and end_id is not None:
        if start_id <= 0 or end_id < start_id:
            raise ValueError("--start-id/--end-id must be a positive inclusive range")
        available = set(discover_formal_response_ids(dataset_path))
        selected = tuple(range(start_id, end_id + 1))
        missing = [sample_id for sample_id in selected if sample_id not in available]
        if missing:
            display = ", ".join(str(sample_id) for sample_id in missing[:10])
            suffix = "" if len(missing) <= 10 else f" ... ({len(missing)} total)"
            raise FileNotFoundError(
                "Requested formal source-ID range is incomplete; missing response IDs: "
                f"{display}{suffix}"
            )
        return FormalSelection(
            sample_ids=selected,
            kind="source_range",
            start_id=start_id,
            end_id=end_id,
        )

    parsed_explicit = parse_sample_ids(explicit_ids)
    if parsed_explicit:
        return FormalSelection(sample_ids=parsed_explicit, kind="explicit")

    defaults = tuple(int(sample_id) for sample_id in configured_ids)
    return FormalSelection(sample_ids=defaults, kind="configured")


def batch_output_root(formal_output_root: Path, start_id: int, end_id: int) -> Path:
    """Return the collision-free default root for one formal source-ID range."""

    selection = FormalSelection(
        sample_ids=tuple(range(start_id, end_id + 1)),
        kind="source_range",
        start_id=start_id,
        end_id=end_id,
    )
    return formal_output_root / "batches" / selection.batch_directory_name
