"""
Graders for Self-RAG answer quality:
  - grade_hallucination: is the answer grounded in the provided documents?
  - grade_utility: does the answer actually address the question?
"""
from pydantic import BaseModel
from openai import OpenAI

from src.config import settings

HALLUCINATION_SYSTEM = """You are a hallucination checker for a RAG system.
Given a set of source documents and a generated answer, determine whether the answer is grounded in the source documents.

Score "yes" if the core claims and key facts in the answer are supported by the documents.
Minor elaborations or general background context are acceptable — focus on whether the main technical assertions are grounded.

Score "no" only if the answer:
- Makes specific technical claims (command syntax, parameter names, behaviour) that directly contradict the documents, OR
- Invents entire sections of content with no basis in the provided documents whatsoever.

Give benefit of the doubt for general framing and minor elaborations."""

UTILITY_SYSTEM = """You are an answer quality evaluator for a PostgreSQL Q&A system.
Given a user question and a generated answer, determine whether the answer is useful.

Score "yes" if the answer:
- Directly addresses the question
- Contains specific technical information (not just vague generalities)
- Would actually help a PostgreSQL user solve their problem

Score "no" if the answer is vague, off-topic, says "I don't know", or fails to address the question."""


class HallucinationResult(BaseModel):
    grounded: str    # "yes" or "no"
    reasoning: str   # 1 sentence


class UtilityResult(BaseModel):
    useful: str      # "yes" or "no"
    reasoning: str   # 1 sentence


def grade_hallucination(docs: list[dict], answer: str) -> HallucinationResult:
    context = "\n\n---\n\n".join(d["text"][:800] for d in docs)
    client = OpenAI(api_key=settings.openai_api_key)
    completion = client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": HALLUCINATION_SYSTEM},
            {
                "role": "user",
                "content": f"Source documents:\n{context}\n\nGenerated answer:\n{answer}",
            },
        ],
        response_format=HallucinationResult,
    )
    return completion.choices[0].message.parsed


def grade_utility(query: str, answer: str) -> UtilityResult:
    client = OpenAI(api_key=settings.openai_api_key)
    completion = client.beta.chat.completions.parse(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": UTILITY_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {query}\n\nAnswer:\n{answer}",
            },
        ],
        response_format=UtilityResult,
    )
    return completion.choices[0].message.parsed
