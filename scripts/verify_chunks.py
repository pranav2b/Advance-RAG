"""
Spot-check chunk quality after extraction.
Run: PYTHONPATH=. python scripts/verify_chunks.py
"""
import json
import random
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def check_file(path: Path, n_samples: int = 5) -> None:
    if not path.exists():
        console.print(f"[red]Missing: {path}[/red]")
        return

    with open(path) as f:
        chunks = json.load(f)

    console.print(f"\n[bold cyan]{path.name}[/bold cyan]: {len(chunks)} chunks")

    table = Table(show_lines=True)
    table.add_column("ID", width=18)
    table.add_column("Source", width=30)
    table.add_column("Page", width=5)
    table.add_column("Text preview (first 200 chars)", width=100)

    samples = random.sample(chunks, min(n_samples, len(chunks)))
    for chunk in samples:
        table.add_row(
            chunk["id"],
            chunk["metadata"].get("source_pdf", ""),
            str(chunk["metadata"].get("page_number", "")),
            chunk["text"][:200].replace("\n", " "),
        )

    console.print(table)

    # Basic stats
    lengths = [len(c["text"]) for c in chunks]
    console.print(
        f"  Char lengths — min: {min(lengths)}, max: {max(lengths)}, avg: {sum(lengths)//len(lengths)}"
    )

    by_source: dict[str, int] = {}
    for c in chunks:
        src = c["metadata"].get("source_pdf", "unknown")
        by_source[src] = by_source.get(src, 0) + 1
    for src, count in sorted(by_source.items()):
        console.print(f"  {src}: {count} chunks")


def main() -> None:
    base = Path("corpus/chunks")
    check_file(base / "naive_chunks.json")
    check_file(base / "contextual_chunks.json")


if __name__ == "__main__":
    main()
