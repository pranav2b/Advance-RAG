"""
Side-by-side comparison of naive vs contextual retrieval.
Teaches: contextual chunks are self-contained — the LLM knows what it's reading.
"""
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from src.config import settings
from src.retrieval.retriever import retrieve

console = Console()


def _split_context_prefix(text: str):
    """
    Extract the context prefix from a contextual chunk.

    Supports two formats:
    1. New 5-element format (single line, pipe-separated):
       'Document: PostgreSQL 17 Documentation | Section: ... | Purpose: ...\n\nchunk body'
    2. Old format (multi-sentence paragraph):
       'PostgreSQL Administration - Chapter...\n\nchunk body'

    Returns (prefix, body). If no prefix found, returns ("", text).
    """
    text = text.strip()

    # Format 1: starts with "Document:" — new 5-element structured prefix
    if text.startswith("Document:"):
        parts = text.split("\n\n", 1)
        if len(parts) == 2:
            return parts[0].strip(), parts[1].strip()
        # No double newline — prefix and body run together on same line
        # Split at last pipe-delimited field boundary
        if "|" in text:
            lines = text.split("\n")
            prefix_lines, body_lines, in_prefix = [], [], True
            for line in lines:
                if in_prefix and (line.startswith("Document:") or "|" in line):
                    prefix_lines.append(line)
                else:
                    in_prefix = False
                    body_lines.append(line)
            if prefix_lines and body_lines:
                return "\n".join(prefix_lines).strip(), "\n".join(body_lines).strip()

    # Format 2: old multi-sentence prefix separated by \n\n
    parts = text.split("\n\n", 1)
    if len(parts) == 2 and len(parts[0]) < 400:
        return parts[0].strip(), parts[1].strip()

    return "", text


def _format_prefix_display(prefix: str) -> str:
    """
    Format the 5-element prefix for readable terminal display.
    Replaces ' | ' with newlines so each element is on its own line.
    """
    if "|" in prefix:
        return prefix.replace(" | ", "\n")
    return prefix


def compare_retrieval(query: str, top_k: int = 3) -> None:
    console.print()
    console.rule("[bold cyan]Session 2 — Contextual Retrieval Comparison[/bold cyan]")
    console.print(f"\n[bold]Query:[/bold] {query}\n")

    naive_results = retrieve(query, settings.collection_naive, top_k=top_k)
    ctx_results = retrieve(query, settings.collection_contextual, top_k=top_k)

    for i in range(top_k):
        console.print(Rule(f"[bold]Rank {i+1}[/bold]", style="dim"))

        # ── NAIVE ──
        if i < len(naive_results):
            r = naive_results[i]
            source = r["metadata"].get("source_pdf", "unknown")
            body = r["text"].strip()[:400].replace("\n", " ")

            naive_text = Text()
            naive_text.append("Source : ", style="dim")
            naive_text.append(source + "\n\n", style="yellow")
            naive_text.append("Chunk text:\n", style="dim italic")
            naive_text.append(body, style="white")

            console.print(
                Panel(
                    naive_text,
                    title="[bold red]NAIVE[/bold red]",
                    border_style="red",
                    padding=(1, 2),
                )
            )

        # ── CONTEXTUAL ──
        if i < len(ctx_results):
            r = ctx_results[i]
            source = r["metadata"].get("source_pdf", "unknown")

            # Prefer stored metadata prefix (new chunker stores it separately)
            stored_prefix = r["metadata"].get("context_prefix", "")
            if stored_prefix:
                prefix = stored_prefix
                raw = r["text"].strip()
                if raw.startswith("Document:"):
                    parts = raw.split("\n\n", 1)
                    body = parts[1].strip() if len(parts) == 2 else raw
                else:
                    body = raw
            else:
                # Fall back to text splitting for old-format chunks
                prefix, body = _split_context_prefix(r["text"])

            body_preview = body[:400].replace("\n", " ")

            ctx_text = Text()
            ctx_text.append("Source : ", style="dim")
            ctx_text.append(source + "\n\n", style="yellow")

            if prefix:
                ctx_text.append("Context prefix (generated):\n", style="dim italic")
                ctx_text.append(_format_prefix_display(prefix) + "\n\n", style="bold cyan")
            else:
                ctx_text.append("Context prefix: ", style="dim italic")
                ctx_text.append("(none detected)\n\n", style="dim")

            ctx_text.append("Chunk text:\n", style="dim italic")
            ctx_text.append(body_preview, style="white")

            console.print(
                Panel(
                    ctx_text,
                    title="[bold green]CONTEXTUAL[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )

        console.print()

    # ── TEACHING POINT ──
    console.rule("[bold cyan]Key Takeaway[/bold cyan]", style="cyan")
    console.print(
        "\n[dim]Naive chunks start mid-sentence — the LLM has no idea which "
        "PostgreSQL subsystem it's reading.\n"
        "Contextual chunks carry a generated prefix that locates the chunk in "
        "the document — the LLM knows exactly what it's reading.[/dim]\n"
    )