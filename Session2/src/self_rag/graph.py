"""
LangGraph Self-RAG graph definition.

Flow:
  retrieve → grade_documents ──→ generate → grade_generation ──→ END (grounded + useful)
                             │                               │
                             └──→ rewrite_query ←───────────┘ (not useful / hallucinated)
                             │         │
                             └──→ web_fallback ←─ (max attempts exceeded)
                                       │
                                       └──→ END
"""
from langgraph.graph import StateGraph, END

from src.self_rag.state import SelfRAGState
from src.self_rag.nodes import (
    MAX_ATTEMPTS,
    node_retrieve,
    node_grade_documents,
    node_generate,
    node_grade_generation,
    node_rewrite_query,
    node_web_fallback,
)


def _route_after_grade_docs(state: SelfRAGState) -> str:
    if state.get("relevant_docs"):
        return "generate"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "web_fallback"
    return "rewrite_query"


def _route_after_grade_generation(state: SelfRAGState) -> str:
    if not state.get("hallucination_detected") and state.get("answer_useful"):
        return "end"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "web_fallback"
    return "rewrite_query"


def build_graph():
    workflow = StateGraph(SelfRAGState)

    workflow.add_node("retrieve", node_retrieve)
    workflow.add_node("grade_documents", node_grade_documents)
    workflow.add_node("generate", node_generate)
    workflow.add_node("grade_generation", node_grade_generation)
    workflow.add_node("rewrite_query", node_rewrite_query)
    workflow.add_node("web_fallback", node_web_fallback)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        _route_after_grade_docs,
        {"generate": "generate", "rewrite_query": "rewrite_query", "web_fallback": "web_fallback"},
    )

    workflow.add_edge("generate", "grade_generation")

    workflow.add_conditional_edges(
        "grade_generation",
        _route_after_grade_generation,
        {"end": END, "rewrite_query": "rewrite_query", "web_fallback": "web_fallback"},
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("web_fallback", END)

    return workflow.compile()


graph = build_graph()


def run(query: str) -> dict:
    """Entry point — invoke the compiled graph and return the final state."""
    initial_state: SelfRAGState = {
        "original_query": query,
        "query": query,
        "documents": [],
        "relevant_docs": [],
        "answer": "",
        "hallucination_detected": False,
        "answer_useful": False,
        "attempts": 0,
        "branch": "grounded_useful",
        "grade_log": [],
    }
    return graph.invoke(initial_state)
