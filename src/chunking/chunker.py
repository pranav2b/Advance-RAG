"""
Naive and contextual chunking of extracted PDFs.
Run:
  PYTHONPATH=. python src/chunking/chunker.py --strategy naive
  PYTHONPATH=. python src/chunking/chunker.py --strategy contextual
"""
import argparse
import json
import time
from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.config import settings

console = Console()

PDF_DIR = Path("corpus/raw")
OUT_DIR = Path("corpus/chunks")
CHUNK_SIZE = 400        # tokens ~ chars * 0.75, use char approximation
CHUNK_OVERLAP = 50
CHAR_PER_TOKEN = 4      # rough approximation

# ── STRONG 5-ELEMENT CONTEXT PROMPT ──────────────────────────────────────────
# Produces measurably better retrieval than 2-3 sentence summaries.
# Targets 25-40 tokens: document title + section + purpose + commands + entities.
CONTEXT_SYSTEM_PROMPT = """You are a technical documentation assistant specialising in enterprise RAG systems.

Given a chunk from the PostgreSQL 17 documentation, generate a structured context prefix containing ALL FIVE of the following elements:

1. Document title  (always: "PostgreSQL 17 Documentation")
2. Section title   (the chapter and subsection this chunk belongs to)
3. Purpose         (one sentence: what this section does or explains)
4. Key concepts, commands, SQL statements, GUC parameters, or APIs discussed — name them explicitly
5. Important entities and relationships — users, roles, objects, actions, errors mentioned

FORMAT your output exactly like this example:
Document: PostgreSQL 17 Documentation | Section: Data Definition — Access Privileges | Purpose: Describes how GRANT statements assign table and column-level privileges to users and roles. Key commands: GRANT, REVOKE, \\dp. Key entities: roles, privileges, ACLs, grantee, grantor.

RULES:
- Target 25-40 tokens — concise but information-dense
- Always name specific PostgreSQL concepts, commands, and parameters — never write vague phrases like "this section covers database concepts"
- If the chunk contains SQL commands or GUC parameters, include them by name in Key commands
- Output the prefix ONLY — no preamble, no explanation, no markdown"""

CONTEXT_USER_TEMPLATE = """Document: {source_pdf}
Chunk text: {chunk_text}

Write only the context prefix. No preamble."""


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """Return list of {page_number, text} dicts from a PDF."""
    doc = fitz.open(str(pdf_path))
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({"page_number": i, "text": text})
    return pages


def chunk_pages(pages: list[dict], source_pdf: str, strategy: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * CHAR_PER_TOKEN,
        chunk_overlap=CHUNK_OVERLAP * CHAR_PER_TOKEN,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = []
    chunk_index = 0
    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            if not split.strip():
                continue
            chunks.append({
                "id": f"{strategy}_{chunk_index:05d}",
                "text": split,
                "metadata": {
                    "source_pdf": source_pdf,
                    "page_number": page["page_number"],
                    "chunk_index": chunk_index,
                    "strategy": strategy,
                },
            })
            chunk_index += 1
    return chunks


def generate_context(client: OpenAI, source_pdf: str, chunk_text: str) -> str:
    """Call GPT-4o-mini to generate a strong 5-element context prefix."""
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": CONTEXT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": CONTEXT_USER_TEMPLATE.format(
                    source_pdf=source_pdf,
                    chunk_text=chunk_text[:1500],
                ),
            },
        ],
        max_tokens=150,
    )
    return response.choices[0].message.content.strip()


def build_naive_chunks() -> list[dict]:
    all_chunks = []
    pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not pdfs:
        console.print("[red]No PDFs found in corpus/raw/. Run extract_pdf.py first.[/red]")
        return []

    for pdf_path in pdfs:
        console.print(f"[cyan]Chunking (naive):[/cyan] {pdf_path.name}")
        pages = extract_text_from_pdf(pdf_path)
        chunks = chunk_pages(pages, pdf_path.name, "naive")
        console.print(f"  → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks


def build_contextual_chunks(auto_confirm: bool = False) -> list[dict]:
    """Load naive chunks and add strong 5-element context prefixes."""
    naive_path = OUT_DIR / "naive_chunks.json"
    if not naive_path.exists():
        console.print("[red]naive_chunks.json not found. Run --strategy naive first.[/red]")
        return []

    with open(naive_path) as f:
        naive_chunks = json.load(f)

    # Cost estimate — strong prompt uses ~120 tokens per call vs ~80 before
    estimated_cost = len(naive_chunks) * 120 * 0.15 / 1_000_000
    console.print(f"\n[yellow]Estimated chunks:[/yellow] {len(naive_chunks)}")
    console.print(f"[yellow]Estimated context generation cost:[/yellow] ~${estimated_cost:.4f}")
    console.print(f"[dim]Using strong 5-element prompt (25-40 token output per chunk)[/dim]")

    if not auto_confirm:
        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            console.print("[red]Aborted.[/red]")
            return []

    client = OpenAI(api_key=settings.openai_api_key)
    contextual_chunks = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Generating context prefixes...", total=len(naive_chunks))

        for i, chunk in enumerate(naive_chunks):
            try:
                prefix = generate_context(
                    client,
                    chunk["metadata"]["source_pdf"],
                    chunk["text"],
                )
            except Exception as e:
                console.print(f"[red]Error on chunk {i}: {e}[/red]")
                prefix = ""

            new_chunk = {
                "id": chunk["id"].replace("naive_", "contextual_"),
                "text": f"{prefix}\n\n{chunk['text']}" if prefix else chunk["text"],
                "metadata": {
                    **chunk["metadata"],
                    "strategy": "contextual",
                    "has_context": bool(prefix),
                    "context_prefix": prefix,   # store separately for inspection
                },
            }
            contextual_chunks.append(new_chunk)
            progress.advance(task)

            # Rate limiting: brief pause per call, longer pause every 50
            if (i + 1) % 50 == 0:
                time.sleep(2)
            else:
                time.sleep(0.1)

    return contextual_chunks


def spot_check_contexts(n: int = 5) -> None:
    """Print n random context prefixes for manual quality check."""
    import random
    ctx_path = OUT_DIR / "contextual_chunks.json"
    if not ctx_path.exists():
        console.print("[red]contextual_chunks.json not found.[/red]")
        return

    with open(ctx_path) as f:
        chunks = json.load(f)

    sample = random.sample(chunks, min(n, len(chunks)))
    console.print(f"\n[bold cyan]Spot-check: {len(sample)} random context prefixes[/bold cyan]\n")
    for i, chunk in enumerate(sample, 1):
        prefix = chunk["metadata"].get("context_prefix", "(none stored)")
        console.print(f"[bold]Chunk {i}[/bold] — {chunk['metadata']['source_pdf']} p{chunk['metadata']['page_number']}")
        console.print(f"  [green]Prefix:[/green] {prefix[:200]}")
        console.print(f"  [dim]Body:[/dim]   {chunk['text'][:120].replace(chr(10), ' ')}...")
        console.print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk PostgreSQL PDF corpus")
    parser.add_argument(
        "--strategy",
        choices=["naive", "contextual", "spot-check"],
        required=True,
        help="naive: split only | contextual: split + context prefix | spot-check: inspect prefixes",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Auto-confirm cost prompt for contextual chunking",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing chunk file instead of skipping",
    )
    args = parser.parse_args()

    if args.strategy == "spot-check":
        spot_check_contexts()
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.strategy == "naive":
        out_path = OUT_DIR / "naive_chunks.json"
        if out_path.exists() and not args.force:
            console.print(f"[yellow]{out_path} already exists — skipping. Use --force to overwrite.[/yellow]")
            with open(out_path) as f:
                chunks = json.load(f)
            console.print(f"[green]Loaded {len(chunks)} existing naive chunks.[/green]")
            return
        chunks = build_naive_chunks()
        if chunks:
            with open(out_path, "w") as f:
                json.dump(chunks, f, indent=2)
            console.print(f"\n[green]Saved {len(chunks)} naive chunks → {out_path}[/green]")

    elif args.strategy == "contextual":
        out_path = OUT_DIR / "contextual_chunks.json"
        if out_path.exists() and not args.force:
            console.print(f"[yellow]{out_path} already exists — skipping. Use --force to overwrite.[/yellow]")
            with open(out_path) as f:
                chunks = json.load(f)
            console.print(f"[green]Loaded {len(chunks)} existing contextual chunks.[/green]")
            return
        chunks = build_contextual_chunks(auto_confirm=args.yes)
        if chunks:
            with open(out_path, "w") as f:
                json.dump(chunks, f, indent=2)
            console.print(f"\n[green]Saved {len(chunks)} contextual chunks → {out_path}[/green]")
            console.print(f"[dim]Run spot-check to verify prefix quality:[/dim]")
            console.print(f"[dim]  PYTHONPATH=. python src/chunking/chunker.py --strategy spot-check[/dim]")


if __name__ == "__main__":
    main()