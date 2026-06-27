# Session 2: Contextual Retrieval + CRAG

PostgreSQL 17 documentation corpus demonstrating two advanced RAG techniques:

1. **Contextual Retrieval** — naive chunks vs GPT-4o-mini context-enriched chunks, side-by-side
2. **CRAG** — three-branch decision loop: all_relevant / partial / web_fallback

## Setup

```bash
# Requires: Qdrant running on localhost:6333, OpenAI API key
cd /Users/pranav/AI/Projects/advanced-rag/session2
uv venv && uv pip install -e .
cp .env.example .env  # fill in OPENAI_API_KEY
```

## Run pipeline (one-time setup)

```bash
# 1. Extract PDF sections
PYTHONPATH=. python src/corpus/extract_pdf.py

# 2. Naive chunks (fast)
PYTHONPATH=. python src/chunking/chunker.py --strategy naive

# 3. Contextual chunks (~$0.006, requires confirmation)
PYTHONPATH=. python src/chunking/chunker.py --strategy contextual

# 4. Index both Qdrant collections
PYTHONPATH=. python src/indexing/indexer.py
```

## Demo commands

### Contextual Retrieval Comparison

```bash
PYTHONPATH=. python demo.py --mode compare --query "How do I grant SELECT privilege on a table to a user?"
PYTHONPATH=. python demo.py --mode compare --query "When should I use a partial index?"
PYTHONPATH=. python demo.py --mode compare --query "What does autovacuum do and when does it trigger?"
```

### CRAG Pipeline — Validated Branch Queries

| Query | Expected Branch | Why |
|-------|----------------|-----|
| "How does role membership and inheritance work in PostgreSQL?" | `all_relevant` | Ch 21 is fully covered; all top-5 chunks address role inheritance |
| "How do I back up a PostgreSQL database and restore it?" | `partial` | Ch 25 backup chunks return but recovery-only chunks are graded irrelevant |
| "What new features were added in PostgreSQL 17 incremental backup?" | `web_fallback` | Grader rejects all corpus chunks as lacking "new features" specifics |

```bash
PYTHONPATH=. python demo.py --mode crag --query "How does role membership and inheritance work in PostgreSQL?"
PYTHONPATH=. python demo.py --mode crag --query "How do I back up a PostgreSQL database and restore it?"
PYTHONPATH=. python demo.py --mode crag --query "What new features were added in PostgreSQL 17 incremental backup?"
```

## Corpus

| File | Content | Pages |
|------|---------|-------|
| `corpus/raw/postgres_data_definition.pdf` | Ch 5: Data Definition (tables, constraints, privileges) | 53 |
| `corpus/raw/postgres_queries_indexes.pdf` | Ch 7: Queries + Ch 11: Indexes + Ch 13: Concurrency | 63 |
| `corpus/raw/postgres_administration.pdf` | Ch 21: Roles + Ch 24: Maintenance + Ch 25: Backup | 36 |

Total: 368 chunks in each collection (`pg_naive`, `pg_contextual`)

## Architecture

```
PDF → extract_pdf.py → corpus/raw/*.pdf
                    → chunker.py (naive)     → naive_chunks.json → pg_naive (Qdrant)
                    → chunker.py (contextual) → contextual_chunks.json → pg_contextual (Qdrant)

Query → retriever.py → Qdrant search
     → compare.py   → side-by-side table
     → crag.py      → grade → branch → generate answer
```

## Models

- Embedding: `text-embedding-3-small` (OpenAI, 1536-dim)
- Context generation: `gpt-4o-mini`
- Grading: `gpt-4o-mini` with structured output (Pydantic)
- Answer generation: `gpt-4o-mini`
