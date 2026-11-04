from src.graph.state import GraphState
from src.config import settings

def route_after_document_grading(state: GraphState) -> str:
    """
    Conditional edge evaluator routing after document grading:
    - If at least one relevant document is found -> route to 'generate'
    - If no relevant documents are found AND retry_count < max_retries -> route to 'rewrite'
    - If no relevant documents are found AND retry_count >= max_retries -> route to 'web_search'
    """
    relevant_docs = state.get("relevant_documents", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", settings.MAX_RETRIES)

    if relevant_docs and len(relevant_docs) > 0:
        print("[Conditional Edge] Found relevant documents -> Routing to 'generator'")
        return "generate"

    if retry_count < max_retries:
        print(f"[Conditional Edge] No relevant docs found. Retries ({retry_count}/{max_retries}) -> Routing to 'query_rewriter'")
        return "rewrite"

    print(f"[Conditional Edge] Max retries reached ({retry_count}/{max_retries}) -> Routing to 'web_search'")
    return "web_search"


def route_after_hallucination_check(state: GraphState) -> str:
    """Only allow answers approved by the evidence auditor to reach the user."""
    if state.get("hallucination_grade") == "grounded":
        return "accept"
    return "reject"
