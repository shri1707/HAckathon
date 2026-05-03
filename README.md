# BIS Standards Recommendation Engine

This project is a proof-of-concept recommendation engine for the **BIS Standards Recommendation Engine Hackathon**. It takes a product description and returns the most relevant Bureau of Indian Standards entries from the BIS SP 21 building-materials dataset.

The system can be used in two ways:

- Backend judge mode through `inference.py`.
- Browser UI mode through `app.py`.

Both modes use the same backend logic in `src/agent.py`.

## What The Application Does

The application reads the BIS SP 21 PDF, extracts standard-summary chunks, embeds them locally, and stores a reusable vector index in `.bis_index/`. For every query, the LangGraph agent performs three steps:

1. Retrieve semantically similar standards from the local embedding index.
2. Ask Gemini 1.5 Flash to choose the best standards from the retrieved candidates.
3. Return ranked BIS standard IDs in the required JSON schema.

If `GEMINI_API_KEY` is not configured or Gemini fails during a run, the app falls back to the local embedding ranking so the judge command and UI still work.

The agent also includes an out-of-scope guard. If a query is unrelated to BIS building-material standards, it returns an empty `retrieved_standards` list with `out_of_scope: true` instead of forcing a nearest-but-wrong recommendation. With Gemini enabled, the app asks Gemini to classify whether the query is within the BIS building-materials context before reranking. Without Gemini, it uses the local retrieval score threshold as a fallback guard.

## Stack

- **LangGraph** for the recommendation workflow.
- **sentence-transformers/all-MiniLM-L6-v2** for local embeddings.
- **Gemini 1.5 Flash** through `google-genai` for optional reranking.
- **FastAPI + Uvicorn** for the web UI/API.
- **pypdf** for BIS SP 21 PDF extraction.

## Project Structure

```text
.
|-- app.py                # FastAPI web UI server
|-- inference.py          # Required judge entrypoint
|-- eval_script.py        # Local public-set evaluator
|-- requirements.txt
|-- README.md
|-- src/
|   |-- agent.py          # LangGraph workflow
|   |-- config.py         # Environment/config handling
|   |-- index.py          # Embedding index creation and search
|   |-- ingest.py         # PDF extraction and chunking
|   |-- llm.py            # Gemini 1.5 Flash reranker
|   |-- retriever.py      # Candidate retrieval
|   `-- text_utils.py     # Standard-id parsing helpers
`-- web/
    |-- index.html        # Browser UI
    |-- styles.css
    `-- app.js
```

## Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the dataset PDF at:

```text
data/dataset.pdf
```

Alternatively, set:

```bash
set BIS_DATASET_PATH=C:\path\to\dataset.pdf
```

If `data/dataset.pdf` is missing, the app also tries:

```text
C:\Users\shri1\Downloads\dataset.pdf
```

Configure Gemini:

```bash
set GEMINI_API_KEY=your_api_key_here
```

Or create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
BIS_USE_GEMINI=1
BIS_GEMINI_MODEL=gemini-1.5-flash
BIS_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

Optional environment variables:

```bash
set BIS_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
set BIS_GEMINI_MODEL=gemini-1.5-flash
set BIS_INDEX_DIR=.bis_index
set BIS_USE_GEMINI=1
```

To force local embedding-only ranking:

```bash
set BIS_USE_GEMINI=0
```

## Required Input Format

The judge input must be a JSON array. Each item must contain `id` and `query`. It may also contain `expected_standards` for local testing.

```json
[
  {
    "id": "PUB-01",
    "query": "We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement. Which BIS standard covers the chemical and physical requirements for our product?"
  }
]
```

## Required Output Format

The generated output follows the sample output style:

```json
[
  {
    "id": "PUB-01",
    "query": "We are a small enterprise manufacturing 33 Grade Ordinary Portland Cement. Which BIS standard covers the chemical and physical requirements for our product?",
    "expected_standards": [
      "IS 269: 1989"
    ],
    "retrieved_standards": [
      "IS 269: 1989",
      "IS 8112: 1989",
      "IS 12269: 1987",
      "IS 455: 1989",
      "IS 1489 (Part 1): 1991"
    ],
    "latency_seconds": 1.24
  }
]
```

For hidden/private inputs that do not include `expected_standards`, the application still preserves `id` and `query`, then writes `retrieved_standards` and `latency_seconds`.

For unrelated queries, the output includes:

```json
{
  "id": "UI-QUERY",
  "query": "Which colour is the sky?",
  "retrieved_standards": [],
  "latency_seconds": 0.05,
  "out_of_scope": true,
  "message": "This query does not appear to describe a building material or BIS standard from the provided dataset."
}
```

## Running Backend Inference

Use the exact command shape from the hackathon rulebook:

```bash
python inference.py --input public_test_set.json --output team_results.json
```

Example local run:

```bash
python inference.py --input "C:\Users\shri1\Downloads\public_test_set.json" --output "data\public_results.json"
```

The first run builds the embedding index and may take longer. Later runs reuse `.bis_index/`.

## Generated Files And Git

The `.bis_index/` directory is generated automatically from `dataset.pdf`. It contains extracted chunks, embeddings, and cache metadata:

```text
.bis_index/chunks.json
.bis_index/embeddings.npy
.bis_index/manifest.json
```

This directory is intentionally ignored by git and does not need to be committed. If `.bis_index/` is missing, the app rebuilds it on the next run as long as `dataset.pdf` is available.

Commit the source files, frontend files, README, requirements, and `.env.example`. Do not commit `.env`, `.bis_index/`, Python cache files, or generated result files such as `data/public_results.json`.

For a clean run on another machine or by judges, make sure the dataset PDF is available at:

```text
data/dataset.pdf
```

or set `BIS_DATASET_PATH` to the dataset location.

## Running The Web UI

Start the server:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Useful API endpoints:

```text
GET  /api/health
POST /api/recommend
```

Example API request:

```json
{
  "id": "UI-QUERY",
  "query": "Which BIS standard applies to white Portland cement?",
  "top_k": 5
}
```

## Local Evaluation

If your input file contains `expected_standards`, evaluate predictions with:

```bash
python eval_script.py --expected public_test_set.json --predictions team_results.json
```

The evaluator prints:

- Hit Rate @3
- MRR @5
- Average latency

## Notes

- The application only returns standards found in the provided BIS dataset candidates.
- Gemini is used as a reranker, not as the source of truth.
- The local fallback keeps the app runnable even without an API key. On the provided public test set, it produced 90.00% Hit Rate @3 and 0.7583 MRR @5 after the index was built.
