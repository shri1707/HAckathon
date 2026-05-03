"""PDF ingestion for BIS SP 21 summaries."""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from .text_utils import attach_split_part_ids, clean_text, find_standard_ids


logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass
class StandardChunk:
    chunk_id: str
    text: str
    standard_ids: list[str]
    page_number: int


def _page_heading(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if "SUMMARY OF" in line.upper():
            return " ".join(lines[index : index + 4])
    return " ".join(lines[:8])


def _chunk_page(page_number: int, text: str) -> list[StandardChunk]:
    text = clean_text(attach_split_part_ids(text))
    if not text:
        return []

    # A page can contain more than one summary. Preserve page context but split
    # when clear "SUMMARY OF" boundaries exist.
    parts = re.split(r"(?=SUMMARY\s+OF\s+IS\s+)", text, flags=re.IGNORECASE)
    chunks = []
    for part_index, part in enumerate(parts):
        part = clean_text(part)
        if len(part) < 80:
            continue
        standard_ids = find_standard_ids(part)
        if not standard_ids:
            standard_ids = find_standard_ids(_page_heading(text))
        if not standard_ids:
            continue
        chunks.append(
            StandardChunk(
                chunk_id=f"p{page_number:04d}-{part_index:02d}",
                text=part,
                standard_ids=standard_ids,
                page_number=page_number,
            )
        )
    return chunks


def load_pdf_chunks(dataset_path: Path) -> list[StandardChunk]:
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset PDF not found at {dataset_path}. Put dataset.pdf in data/ "
            "or set BIS_DATASET_PATH."
        )

    reader = PdfReader(str(dataset_path))
    chunks: list[StandardChunk] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.extend(_chunk_page(page_index, text))
    if not chunks:
        raise RuntimeError("No standard chunks could be extracted from the dataset PDF.")
    return chunks
