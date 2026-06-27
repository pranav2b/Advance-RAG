"""
Query rewriter for Self-RAG retry loop.
When the generated answer is not grounded or not useful, rewrites the query
to be more specific and retrieval-friendly.
"""
from openai import OpenAI

from src.config import settings

REWRITE_SYSTEM = """You are a query optimisation expert for a PostgreSQL documentation retrieval system.
When a retrieval query fails to return useful results, you rewrite it to be more specific and targeted.

Rules:
- Add specific PostgreSQL terminology (command names, GUC parameters, chapter topics)
- Break compound questions into the most specific sub-question
- If the previous answer was vague or wrong, steer toward the precise concept
- Output ONLY the rewritten query — no explanation, no preamble"""


def rewrite_query(original_query: str, failed_answer: str) -> str:
    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Original query: {original_query}\n\n"
                    f"Previous answer (unsatisfactory):\n{failed_answer[:500]}\n\n"
                    "Rewrite the query to retrieve better documentation chunks:"
                ),
            },
        ],
        max_tokens=80,
    )
    return response.choices[0].message.content.strip()
