"""
LangGraph node functions for Self-RAG pipeline.
Each function receives the full state and returns a dict with updated keys.
"""
from rich.console import Console

from src.config import settings
from src.retrieval.retriever import retrieve
from src.crag.grader import grade_document
from src.crag.web_search import web_search_fallback
from src.self_rag.state import SelfRAGState
from src.self_rag.grade_generation import grade_hallucination, grade_utility
from src.self_rag.query_rewriter import rewrite_query
from openai import OpenAI

console = Console()

ANSWER_SYSTEM = """You are a PostgreSQL expert assistant. Answer the question using ONLY the provided context.
Be specific and technical. Cite which section of the documentation your answer comes from.
If you cannot answer from the context, say so clearly — do not invent information."""

MAX_ATTEMPTS = 3


# ── Node: retrieve ────────────────────────────────────────────────────────────

def node_retrieve(state: SelfRAGState) -> dict:
    query = state["query"]
    attempts = state.get("attempts", 0)
    console.print(f"\n[bold cyan][RETRIEVE][/bold cyan] Attempt {attempts + 1}/{MAX_ATTEMPTS} — query: \"{query}\"")

    docs = retrieve(query, settings.collection_contextual, top_k=5)
    console.print(f"[dim]  → Retrieved {len(docs)} chunks from '{settings.collection_contextual}'[/dim]")

    return {
        "documents": docs,
        "attempts": attempts + 1,
    }


# ── Node: grade_documents ─────────────────────────────────────────────────────

def node_grade_documents(state: SelfRAGState) -> dict:
    query = state["query"]
    docs = state["documents"]
    grade_log = state.get("grade_log", [])

    console.print(f"\n[bold magenta][GRADE DOCS][/bold magenta] Grading {len(docs)} chunks for relevance...")

    relevant_docs = []
    for i, doc in enumerate(docs, 1):
        preview = doc["text"][:80].replace("\n", " ")
        result = grade_document(query, doc["text"])
        color = "green" if result.score == "relevant" else "red"
        label = "RELEVANT  " if result.score == "relevant" else "IRRELEVANT"
        console.print(f"  [{color}][{i}/5] {label}[/{color}] — \"{preview}...\"")
        console.print(f"  [dim]{result.reasoning}[/dim]")

        grade_log.append({
            "stage": "doc_grading",
            "attempt": state.get("attempts", 1),
            "doc_preview": preview,
            "score": result.score,
            "reasoning": result.reasoning,
        })
        if result.score == "relevant":
            relevant_docs.append(doc)

    console.print(f"\n  → [bold]{len(relevant_docs)}/{len(docs)} relevant[/bold]")
    return {
        "relevant_docs": relevant_docs,
        "grade_log": grade_log,
    }


# ── Node: generate ────────────────────────────────────────────────────────────

def node_generate(state: SelfRAGState) -> dict:
    query = state["query"]
    relevant_docs = state.get("relevant_docs", [])
    docs_to_use = relevant_docs if relevant_docs else state.get("documents", [])

    console.print(f"\n[bold yellow][GENERATE][/bold yellow] Generating answer from {len(docs_to_use)} docs...")

    context = "\n\n---\n\n".join(d["text"] for d in docs_to_use)
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
        ],
        max_tokens=500,
    )
    answer = response.choices[0].message.content.strip()
    console.print(f"[dim]  → Answer generated ({len(answer)} chars)[/dim]")

    return {"answer": answer}


# ── Node: grade_generation ────────────────────────────────────────────────────

def node_grade_generation(state: SelfRAGState) -> dict:
    query = state["original_query"]
    answer = state["answer"]
    docs = state.get("relevant_docs") or state.get("documents", [])
    grade_log = state.get("grade_log", [])

    console.print(f"\n[bold blue][GRADE GEN][/bold blue] Checking answer quality...")

    # Hallucination check
    hall_result = grade_hallucination(docs, answer)
    color = "green" if hall_result.grounded == "yes" else "red"
    console.print(f"  Grounded:  [{color}]{hall_result.grounded.upper()}[/{color}] — {hall_result.reasoning}")

    # Utility check
    util_result = grade_utility(query, answer)
    color = "green" if util_result.useful == "yes" else "red"
    console.print(f"  Useful:    [{color}]{util_result.useful.upper()}[/{color}] — {util_result.reasoning}")

    grade_log.append({
        "stage": "generation_grading",
        "attempt": state.get("attempts", 1),
        "grounded": hall_result.grounded,
        "hallucination_reasoning": hall_result.reasoning,
        "useful": util_result.useful,
        "utility_reasoning": util_result.reasoning,
    })

    return {
        "hallucination_detected": hall_result.grounded == "no",
        "answer_useful": util_result.useful == "yes",
        "grade_log": grade_log,
    }


# ── Node: rewrite_query ───────────────────────────────────────────────────────

def node_rewrite_query(state: SelfRAGState) -> dict:
    original = state["query"]
    answer = state["answer"]
    console.print(f"\n[bold orange3][REWRITE][/bold orange3] Answer not useful — rewriting query...")

    new_query = rewrite_query(original, answer)
    console.print(f"  Original: \"{original}\"")
    console.print(f"  Rewritten: \"{new_query}\"")

    return {
        "query": new_query,
        "branch": "query_rewrite",
    }


# ── Node: web_fallback ────────────────────────────────────────────────────────

def node_web_fallback(state: SelfRAGState) -> dict:
    query = state["original_query"]
    console.print(f"\n[bold yellow][WEB FALLBACK][/bold yellow] Max retries hit — falling back to web search...")

    result = web_search_fallback(query)
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": "You are a PostgreSQL expert. Answer using the web search result provided."},
            {"role": "user", "content": f"Question: {query}\n\nWeb search result:\n{result}"},
        ],
        max_tokens=400,
    )
    answer = response.choices[0].message.content.strip()

    return {
        "answer": answer,
        "branch": "web_fallback",
    }
