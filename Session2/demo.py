"""
Session 2 demo entry point — contextual retrieval + CRAG + Self-RAG.

Usage:
  PYTHONPATH=. python demo.py --mode compare --query "How do I grant SELECT privilege?"
  PYTHONPATH=. python demo.py --mode crag    --query "What is the difference between VACUUM and AUTOVACUUM?"
  PYTHONPATH=. python demo.py --mode self_rag --query "How does role membership and inheritance work in PostgreSQL?"
"""
import argparse

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

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


def run_self_rag(query: str) -> None:
    console.print(Rule("[bold green]Session 2 — Self-RAG Pipeline (LangGraph)[/bold green]"))

    from src.self_rag.graph import run
    state = run(query)

    # ── Answer ──
    console.print()
    console.print(Rule("[bold]Answer[/bold]"))
    console.print(Panel(state["answer"], title="[bold green]Generated Answer[/bold green]", border_style="green"))

    # ── Summary ──
    branch = state.get("branch", "grounded_useful")
    branch_color = {"grounded_useful": "green", "query_rewrite": "blue", "web_fallback": "yellow"}.get(branch, "white")
    console.print(f"\n[bold]Branch:[/bold]   [{branch_color}]{branch}[/{branch_color}]")
    console.print(f"[bold]Attempts:[/bold] {state.get('attempts', 0)}")

    # ── Grade log ──
    log = state.get("grade_log", [])
    if log:
        console.print()
        console.print(Rule("[bold]Grade Log[/bold]", style="dim"))
        table = Table(show_lines=True, expand=False)
        table.add_column("Stage", width=18)
        table.add_column("Attempt", width=7)
        table.add_column("Result", width=12)
        table.add_column("Reasoning", width=70)

        for entry in log:
            if entry["stage"] == "doc_grading":
                color = "green" if entry["score"] == "relevant" else "red"
                table.add_row(
                    "doc_grading",
                    str(entry["attempt"]),
                    f"[{color}]{entry['score']}[/{color}]",
                    entry["reasoning"][:68],
                )
            elif entry["stage"] == "generation_grading":
                grounded_color = "green" if entry["grounded"] == "yes" else "red"
                useful_color = "green" if entry["useful"] == "yes" else "red"
                table.add_row(
                    "hallucination",
                    str(entry["attempt"]),
                    f"[{grounded_color}]grounded={entry['grounded']}[/{grounded_color}]",
                    entry["hallucination_reasoning"][:68],
                )
                table.add_row(
                    "utility",
                    str(entry["attempt"]),
                    f"[{useful_color}]useful={entry['useful']}[/{useful_color}]",
                    entry["utility_reasoning"][:68],
                )
        console.print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Session 2 — Contextual RAG + CRAG + Self-RAG Demo")
    parser.add_argument(
        "--mode",
        choices=["compare", "crag", "self_rag"],
        required=True,
        help="Demo mode",
    )
    parser.add_argument("--query", required=True, help="Question to answer")
    args = parser.parse_args()

    if args.mode == "compare":
        run_compare(args.query)
    elif args.mode == "crag":
        run_crag(args.query)
    elif args.mode == "self_rag":
        run_self_rag(args.query)


if __name__ == "__main__":
    main()
