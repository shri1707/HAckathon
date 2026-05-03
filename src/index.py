"""Embedding index backed by all-MiniLM-L6-v2 sentence embeddings."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .ingest import StandardChunk, load_pdf_chunks


class VectorIndex:
    def __init__(self, model_name: str, index_dir: Path, dataset_path: Path) -> None:
        self.model_name = model_name
        self.index_dir = index_dir
        self.dataset_path = dataset_path
        try:
            self.model = SentenceTransformer(model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(model_name)
        self.chunks: list[StandardChunk] = []
        self.embeddings: np.ndarray | None = None

    @property
    def metadata_path(self) -> Path:
        return self.index_dir / "chunks.json"

    @property
    def embeddings_path(self) -> Path:
        return self.index_dir / "embeddings.npy"

    @property
    def manifest_path(self) -> Path:
        return self.index_dir / "manifest.json"

    def load_or_build(self) -> None:
        if self._is_cache_valid():
            self._load()
            return
        self._build()

    def search(self, query: str, top_n: int = 24) -> list[tuple[StandardChunk, float]]:
        if self.embeddings is None:
            raise RuntimeError("Vector index is not loaded.")
        query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
        scores = np.dot(self.embeddings, query_embedding)
        top_indices = np.argsort(scores)[::-1][:top_n]
        return [(self.chunks[int(index)], float(scores[int(index)])) for index in top_indices]

    def _is_cache_valid(self) -> bool:
        if not (self.metadata_path.exists() and self.embeddings_path.exists() and self.manifest_path.exists()):
            return False
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return (
            manifest.get("dataset_path") == str(self.dataset_path.resolve())
            and manifest.get("dataset_mtime") == self.dataset_path.stat().st_mtime
            and manifest.get("model_name") == self.model_name
        )

    def _load(self) -> None:
        raw_chunks = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.chunks = [StandardChunk(**item) for item in raw_chunks]
        self.embeddings = np.load(self.embeddings_path)

    def _build(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks = load_pdf_chunks(self.dataset_path)
        texts = [self._embedding_text(chunk) for chunk in self.chunks]
        vectors = []
        for start in tqdm(range(0, len(texts), 64), desc="Embedding BIS chunks"):
            batch = texts[start : start + 64]
            vectors.append(self.model.encode(batch, normalize_embeddings=True))
        self.embeddings = np.vstack(vectors).astype("float32")

        self.metadata_path.write_text(
            json.dumps([asdict(chunk) for chunk in self.chunks], ensure_ascii=False),
            encoding="utf-8",
        )
        np.save(self.embeddings_path, self.embeddings)
        self.manifest_path.write_text(
            json.dumps(
                {
                    "dataset_path": str(self.dataset_path.resolve()),
                    "dataset_mtime": self.dataset_path.stat().st_mtime,
                    "model_name": self.model_name,
                    "chunk_count": len(self.chunks),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _embedding_text(chunk: StandardChunk) -> str:
        ids = ", ".join(chunk.standard_ids)
        return f"{ids}\nPage {chunk.page_number}\n{chunk.text}"
