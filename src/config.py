"""Configuration helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    dataset_path: Path
    index_dir: Path
    embedding_model: str
    gemini_model: str
    gemini_api_key: str | None
    use_gemini: bool

    @classmethod
    def from_env(cls) -> "Settings":
        repo_root = Path(__file__).resolve().parents[1]
        load_dotenv(repo_root / ".env")

        fallback_download = Path.home() / "Downloads" / "dataset.pdf"
        dataset_path = Path(
            os.getenv("BIS_DATASET_PATH", repo_root / "data" / "dataset.pdf")
        )
        if not dataset_path.exists() and fallback_download.exists():
            dataset_path = fallback_download

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if api_key in {"", "put_your_gemini_api_key_here", "your_api_key_here"}:
            api_key = None
        use_gemini = os.getenv("BIS_USE_GEMINI", "1").lower() not in {"0", "false", "no"}

        return cls(
            dataset_path=dataset_path,
            index_dir=Path(os.getenv("BIS_INDEX_DIR", repo_root / ".bis_index")),
            embedding_model=os.getenv("BIS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
            gemini_model=os.getenv("BIS_GEMINI_MODEL", "gemini-1.5-flash"),
            gemini_api_key=api_key,
            use_gemini=use_gemini,
        )
