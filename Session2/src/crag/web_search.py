"""
DuckDuckGo web search fallback for CRAG pipeline.
"""
from langchain_community.tools import DuckDuckGoSearchRun


def web_search_fallback(query: str) -> str:
    """Search DuckDuckGo with PostgreSQL appended. Returns result text or empty string."""
    search = DuckDuckGoSearchRun()
    try:
        result = search.run(f"{query} PostgreSQL")
        return result
    except Exception as e:
        return f"[Web search unavailable: {e}]"
