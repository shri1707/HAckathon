"""Text normalization and standard-id helpers."""

from __future__ import annotations

import re


STANDARD_PATTERN = re.compile(
    r"\bIS\s+([0-9]{2,5})\s*"
    r"(?:\(\s*(PART|Pt|P)\s*[- ]?\s*([0-9A-Z]+)\s*\))?\s*"
    r"[:\-]\s*([0-9]{4})\b",
    re.IGNORECASE,
)

LOOSE_PART_PATTERN = re.compile(
    r"\(Part\s+([0-9A-Z]+)\)\s*:\s*([0-9]{4})",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def canonical_standard_id(raw: str) -> str | None:
    match = STANDARD_PATTERN.search(raw)
    if not match:
        return None

    number, _part_label, part_number, year = match.groups()
    if part_number:
        return f"IS {int(number)} (Part {part_number.upper()}): {year}"
    return f"IS {int(number)}: {year}"


def find_standard_ids(text: str) -> list[str]:
    ids = []
    seen = set()
    for match in STANDARD_PATTERN.finditer(text):
        standard_id = canonical_standard_id(match.group(0))
        if standard_id and standard_id not in seen:
            seen.add(standard_id)
            ids.append(standard_id)
    return ids


def attach_split_part_ids(text: str) -> str:
    """Normalize TOC text like 'IS 1489 : ... (Part 2) : 1991' for matching."""
    current_number = None
    output = []
    for line in text.splitlines():
        number_match = re.search(r"\bIS\s+([0-9]{2,5})\b", line, re.IGNORECASE)
        if number_match:
            current_number = number_match.group(1)
        if current_number and re.search(r"^\s*\(Part\s+", line, re.IGNORECASE):
            line = f"IS {current_number} {line}"
        output.append(line)
    return "\n".join(output)


def compact_for_prompt(text: str, max_chars: int = 900) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
