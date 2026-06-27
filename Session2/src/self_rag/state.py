from typing import TypedDict


class SelfRAGState(TypedDict):
    original_query: str
    query: str            # current query — may be rewritten across attempts
    documents: list       # all retrieved docs (current attempt)
    relevant_docs: list   # filtered by doc grader
    answer: str
    hallucination_detected: bool
    answer_useful: bool
    attempts: int
    branch: str           # grounded_useful | query_rewrite | web_fallback
    grade_log: list       # audit trail of all grading decisions
