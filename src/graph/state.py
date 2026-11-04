from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class DocumentChunk(TypedDict, total=False):
    content: str
    metadata: Dict[str, Any]
    relevance_score: Optional[str]  # "relevant" | "irrelevant"
    reasoning: Optional[str]

class GraphState(TypedDict, total=False):
    """
    Central state schema flowing through the LangGraph self-corrective workflow.
    """
    question: str                      # Original raw user query
    current_query: str                 # Active query (rewritten or expanded)
    query_type: Optional[str]          # conceptual | how-to | troubleshooting | api_reference
    chat_history: List[Dict[str, str]] # Prior conversation turns
    documents: List[DocumentChunk]      # Candidate retrieved documents
    relevant_documents: List[DocumentChunk] # Chunks that passed document grading
    generation: Optional[str]          # Generated final response
    sources: List[Dict[str, Any]]      # Citations list (source, header_path, snippet)
    retry_count: int                   # Number of rewrites executed
    max_retries: int                   # Maximum permitted rewrite retries
    web_search_needed: bool            # True if retrieval failed and web search should be used
    hallucination_grade: Optional[str] # "grounded" | "not grounded"
    verification_reason: Optional[str]
    execution_steps: List[str]         # Audit trail of executed nodes for observability
