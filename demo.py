"""
Session 2 demo entry point — contextual retrieval + CRAG.

Usage:
  PYTHONPATH=. python demo.py --mode compare --query "How do I grant SELECT privilege?"
  PYTHONPATH=. python demo.py --mode crag --query "What is the difference between VACUUM and AUTOVACUUM?"
  PYTHONPATH=. python demo.py --mode crag --query "What new features were added in PostgreSQL 17 incremental backup?"
"""
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule

from src.compare.compare import compare_retrieval
from src.crag.crag import CRAGPipeline

console = Console()


def run_compare(query: str) -> None:
    console.print(Rule("[bold blue]Session 2 — Contextual Retrieval Comparison[/bold blue]"))
    compare_retrieval(query, top_k=3)


def run_crag(query: str) -> None:
    console.print(Rule("[bold magenta]Session 2 — CRAG Pipeline[/bold magenta]"))
    pipeline = CRAGPipeline()
    result = pipeline.run(query)

    console.print()
    console.print(Rule("[bold]Answer[/bold]"))
    console.print(Panel(result["answer"], title="[bold green]Generated Answer[/bold green]", border_style="green"))

    console.print(f"\n[bold]Branch:[/bold]        [cyan]{result['branch']}[/cyan]")
    console.print(f"[bold]Docs retrieved:[/bold] {result['docs_retrieved']}")
    console.print(f"[bold]Docs used:[/bold]      {result['docs_used']}")
    console.print(f"[bold]Sources:[/bold]        {', '.join(result['sources']) or 'web search'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Session 2 — Contextual RAG + CRAG Demo")
    parser.add_argument(
        "--mode",
        choices=["compare", "crag"],
        required=True,
        help="Demo mode: 'compare' shows naive vs contextual, 'crag' runs three-branch pipeline",
    )
    parser.add_argument("--query", required=True, help="Question to answer")
    args = parser.parse_args()

    if args.mode == "compare":
        run_compare(args.query)
    elif args.mode == "crag":
        run_crag(args.query)


if __name__ == "__main__":
    main()
