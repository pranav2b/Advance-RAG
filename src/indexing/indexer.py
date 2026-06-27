"""
Embed chunks and store in Qdrant.
Run: PYTHONPATH=. python src/indexing/indexer.py
"""
import json
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from src.config import settings

console = Console()
BATCH_SIZE = 100


def get_client() -> QdrantClient:
    return QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)


def create_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        client.delete_collection(name)
        console.print(f"[yellow]Deleted existing collection: {name}[/yellow]")
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
    )
    console.print(f"[green]Created collection: {name}[/green]")


def embed_texts(oai_client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = oai_client.embeddings.create(
        model=settings.embedding_model,
        input=texts,
    )
    return [item.embedding for item in response.data]


def embed_and_upsert(
    oai_client: OpenAI,
    qdrant_client: QdrantClient,
    chunks: list[dict],
    collection_name: str,
) -> None:
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Indexing into '{collection_name}'...", total=len(chunks))

        for batch_start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[batch_start : batch_start + BATCH_SIZE]
            texts = [c["text"] for c in batch]
            embeddings = embed_texts(oai_client, texts)

            points = [
                PointStruct(
                    id=batch_start + i,
                    vector=embedding,
                    payload={**chunk["metadata"], "text": chunk["text"], "chunk_id": chunk["id"]},
                )
                for i, (chunk, embedding) in enumerate(zip(batch, embeddings))
            ]
            qdrant_client.upsert(collection_name=collection_name, points=points)
            progress.advance(task, len(batch))

    info = qdrant_client.get_collection(collection_name)
    console.print(
        f"[green]Collection '{collection_name}': {info.points_count} points indexed[/green]"
    )


def load_chunks(path: Path) -> list[dict]:
    if not path.exists():
        console.print(f"[red]File not found: {path}[/red]")
        return []
    with open(path) as f:
        return json.load(f)


def main() -> None:
    oai_client = OpenAI(api_key=settings.openai_api_key)
    qdrant_client = get_client()

    # Verify Qdrant connection
    try:
        qdrant_client.get_collections()
        console.print(f"[green]Connected to Qdrant at {settings.qdrant_host}:{settings.qdrant_port}[/green]")
    except Exception as e:
        console.print(f"[red]Cannot connect to Qdrant: {e}[/red]")
        return

    naive_chunks = load_chunks(Path("corpus/chunks/naive_chunks.json"))
    contextual_chunks = load_chunks(Path("corpus/chunks/contextual_chunks.json"))

    if not naive_chunks or not contextual_chunks:
        console.print("[red]Missing chunk files. Run chunker.py first.[/red]")
        return

    console.print(f"\nNaive chunks: {len(naive_chunks)}")
    console.print(f"Contextual chunks: {len(contextual_chunks)}")

    # Index naive
    console.print(f"\n[bold]Indexing naive chunks → '{settings.collection_naive}'[/bold]")
    create_collection(qdrant_client, settings.collection_naive)
    embed_and_upsert(oai_client, qdrant_client, naive_chunks, settings.collection_naive)

    # Index contextual
    console.print(f"\n[bold]Indexing contextual chunks → '{settings.collection_contextual}'[/bold]")
    create_collection(qdrant_client, settings.collection_contextual)
    embed_and_upsert(oai_client, qdrant_client, contextual_chunks, settings.collection_contextual)

    # Verify both have same point count
    naive_info = qdrant_client.get_collection(settings.collection_naive)
    ctx_info = qdrant_client.get_collection(settings.collection_contextual)
    if naive_info.points_count == ctx_info.points_count:
        console.print(
            f"\n[green]Both collections have {naive_info.points_count} points. Indexing complete.[/green]"
        )
    else:
        console.print(
            f"\n[yellow]Point count mismatch: {settings.collection_naive}={naive_info.points_count}, "
            f"{settings.collection_contextual}={ctx_info.points_count}[/yellow]"
        )


if __name__ == "__main__":
    main()
