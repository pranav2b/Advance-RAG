"""
Relevance grader for CRAG pipeline.
"""
from pydantic import BaseModel
from openai import OpenAI

from src.config import settings

SYSTEM_PROMPT = """You are a relevance grader for a PostgreSQL documentation retrieval system.
Given a user question and a retrieved document chunk, determine if the chunk contains information useful for answering the question.

Score "relevant" if the chunk contains directly useful information about the topic, even if it doesn't cover every aspect of the question.
Score "irrelevant" only if the chunk is clearly off-topic or from the wrong subsystem entirely.
Partial coverage of the question still counts as relevant."""


class GradeResult(BaseModel):
    score: str       # "relevant" or "irrelevant"
    reasoning: str   # 1 sentence explanation


def grade_document(query: str, document: str) -> GradeResult:
    """Grade whether document is relevant to query using structured output."""
    client = OpenAI(api_key=settings.openai_api_key)
    completion = client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Question: {query}\n\nDocument chunk:\n{document[:1500]}",
            },
        ],
        response_format=GradeResult,
    )
    return completion.choices[0].message.parsed