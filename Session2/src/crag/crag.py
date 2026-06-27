"""
Three-branch CRAG pipeline: all_relevant / partial / web_fallback.
"""
from openai import OpenAI
from rich.console import Console

from src.config import settings
from src.crag.grader import grade_document
from src.crag.web_search import web_search_fallback
from src.retrieval.retriever import retrieve

console = Console()

ANSWER_SYSTEM_PROMPT = """You are a PostgreSQL expert assistant. Answer the question using ONLY the provided context. Be specific and technical. If the context is from web search, say so. Cite which section of the documentation your answer comes from."""

ANSWER_USER_TEMPLATE = """Question: {query}

Context:
{context}"""


def _generate_answer(query: str, context: str) -> str:
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": ANSWER_USER_TEMPLATE.format(query=query, context=context)},
        ],
        max_tokens=500,
    )
    return response.choices[0].message.content.strip()


class CRAGPipeline:
    def run(self, query: str) -> dict:
        """
        Three-branch CRAG:
          ALL irrelevant → web_fallback
          1-4 relevant   → partial
          ALL relevant   → all_relevant
        """
        console.print(f"\n[bold cyan][RETRIEVAL][/bold cyan] Fetching top-5 chunks from '{settings.collection_contextual}'...")
        docs = retrieve(query, settings.collection_contextual, top_k=5)

        grading_results = []
        relevant_docs = []

        for i, doc in enumerate(docs, 1):
            preview = doc["text"][:80].replace("\n", " ")
            grade = grade_document(query, doc["text"])

            color = "green" if grade.score == "relevant" else "red"
            label = "RELEVANT  " if grade.score == "relevant" else "IRRELEVANT"
            console.print(
                f"[bold {color}][GRADE {i}/5][/bold {color}] {label} — \"{preview}...\""
            )
            console.print(f"           [dim]{grade.reasoning}[/dim]")

            grading_results.append({
                "text_preview": preview,
                "score": grade.score,
                "reasoning": grade.reasoning,
                "source": doc["metadata"].get("source_pdf", ""),
            })
            if grade.score == "relevant":
                relevant_docs.append(doc)

        n_relevant = len(relevant_docs)
        n_total = len(docs)

        if n_relevant == 0:
            branch = "web_fallback"
            console.print(f"\n[bold yellow][BRANCH][/bold yellow]    WEB FALLBACK — 0/{n_total} chunks relevant, searching web...")
            web_result = web_search_fallback(query)
            context = f"[Web search result]\n{web_result}"
            sources = ["DuckDuckGo web search"]
        elif n_relevant == n_total:
            branch = "all_relevant"
            console.print(f"\n[bold green][BRANCH][/bold green]    ALL RELEVANT — {n_relevant}/{n_total} chunks relevant, using all")
            context = "\n\n---\n\n".join(d["text"] for d in docs)
            sources = list({d["metadata"].get("source_pdf", "") for d in docs})
        else:
            branch = "partial"
            console.print(f"\n[bold blue][BRANCH][/bold blue]    PARTIAL — {n_relevant}/{n_total} chunks relevant, using filtered set")
            context = "\n\n---\n\n".join(d["text"] for d in relevant_docs)
            sources = list({d["metadata"].get("source_pdf", "") for d in relevant_docs})

        console.print(f"[bold cyan][ANSWER][/bold cyan]    Generating with {settings.llm_model}...")
        answer = _generate_answer(query, context)

        return {
            "query": query,
            "branch": branch,
            "docs_retrieved": n_total,
            "docs_used": n_relevant if branch != "web_fallback" else 0,
            "grading_results": grading_results,
            "answer": answer,
            "sources": sources,
        }
