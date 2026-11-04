from typing import List
from src.graph.state import GraphState, DocumentChunk
from src.config import settings

def web_search_node(state: GraphState) -> dict:
    """
    Fallback search node executed when vector retrieval retries are exhausted.
    Uses DuckDuckGo (free, no API key needed) or Tavily if configured.
    """
    query = state.get("current_query") or state.get("question", "")
    steps = state.get("execution_steps", []).copy()
    steps.append("web_search_fallback")

    print(f"[Web Search Fallback Node] Executing web search for: '{query}'")
    search_docs: List[DocumentChunk] = []

    try:
        if settings.SEARCH_PROVIDER == "tavily" and settings.TAVILY_API_KEY:
            from tavily import TavilyClient
            client = TavilyClient(api_key=settings.TAVILY_API_KEY)
            response = client.search(query=query, max_results=3)
            for res in response.get("results", []):
                search_docs.append({
                    "content": res.get("content", ""),
                    "metadata": {
                        "source": res.get("url", "web_search"),
                        "title": res.get("title", "Web Search Result"),
                        "header_path": "Web Search Fallback",
                        "source_type": "web",
                    },
                    "relevance_score": "relevant",
                    "reasoning": "Retrieved via Tavily Web Search fallback",
                })
        else:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=3))
                for res in results:
                    search_docs.append({
                        "content": res.get("body", ""),
                        "metadata": {
                            "source": res.get("href", "web_search"),
                            "title": res.get("title", "DuckDuckGo Result"),
                            "header_path": "Web Search Fallback",
                            "source_type": "web",
                        },
                        "relevance_score": "relevant",
                        "reasoning": "Retrieved via DuckDuckGo fallback",
                    })
    except Exception as e:
        print(f"[Web Search Fallback] Search failed or rate limited: {e}")

    print(f"[Web Search Fallback Node] Retrieved {len(search_docs)} external chunks.")

    # Search output is untrusted. Run it through the same document grader
    # before allowing it into the generation context.
    from src.graph.nodes.document_grader import document_grader_node

    graded = document_grader_node({
        **state,
        "documents": search_docs,
        "execution_steps": steps,
    })

    return {
        "relevant_documents": graded.get("relevant_documents", []),
        "web_search_needed": True,
        "execution_steps": graded.get("execution_steps", steps),
    }
