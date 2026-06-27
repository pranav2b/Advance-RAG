"""
Dense retrieval from Qdrant collections.
Run: PYTHONPATH=. python src/retrieval/retriever.py
"""
from openai import OpenAI
from qdrant_client import QdrantClient
from rich.console import Console
from rich.table import Table

from src.config import settings

console = Console()


def _embed_query(client: OpenAI, query: str) -> list[float]:
    response = client.embeddings.create(model=settings.embedding_model, input=[query])
    return response.data[0].embedding


def retrieve(query: str, collection: str, top_k: int = 5) -> list[dict]:
    """Embed query and search Qdrant collection. Returns list of {text, score, metadata}."""
    oai_client = OpenAI(api_key=settings.openai_api_key)
    qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    vector = _embed_query(oai_client, query)
    response = qdrant_client.query_points(
        collection_name=collection,
        query=vector,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "text": hit.payload.get("text", ""),
            "score": round(hit.score, 4),
            "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
        }
        for hit in response.points
    ]


def _print_results(query: str, results: list[dict], collection: str) -> None:
    table = Table(title=f"Top-{len(results)} from '{collection}'", show_lines=True)
    table.add_column("Rank", style="bold", width=4)
    table.add_column("Score", width=6)
    table.add_column("Source", width=30)
    table.add_column("Preview", width=80)

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            str(r["score"]),
            r["metadata"].get("source_pdf", ""),
            r["text"][:120].replace("\n", " "),
        )
    console.print(table)


def main() -> None:
    test_query = "How do I grant SELECT privilege on a table to a user?"
    console.print(f"\n[bold]Test query:[/bold] {test_query}\n")

    for collection in [settings.collection_naive, settings.collection_contextual]:
        try:
            results = retrieve(test_query, collection)
            _print_results(test_query, results, collection)
        except Exception as e:
            console.print(f"[red]Error querying '{collection}': {e}[/red]")


if __name__ == "__main__":
    main()
