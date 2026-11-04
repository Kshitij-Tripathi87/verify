from langgraph.graph import StateGraph, START, END
from src.graph.state import GraphState
from src.graph.nodes.query_analysis import query_analysis_node
from src.graph.nodes.retriever import retriever_node
from src.graph.nodes.document_grader import document_grader_node
from src.graph.nodes.query_rewriter import query_rewriter_node
from src.graph.nodes.web_search import web_search_node
from src.graph.nodes.generator import generator_node
from src.graph.nodes.hallucination import hallucination_check_node
from src.graph.edges import route_after_document_grading, route_after_hallucination_check
from src.graph.nodes.safe_abstention import safe_abstention_node

def create_rag_graph():
    """
    Constructs and compiles the Self-Corrective RAG LangGraph workflow.
    """
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("query_analysis", query_analysis_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("document_grader", document_grader_node)
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("hallucination_check", hallucination_check_node)
    workflow.add_node("safe_abstention", safe_abstention_node)

    # Add Direct Edges
    workflow.add_edge(START, "query_analysis")
    workflow.add_edge("query_analysis", "retriever")
    workflow.add_edge("retriever", "document_grader")

    # Add Self-Corrective Conditional Edge
    workflow.add_conditional_edges(
        "document_grader",
        route_after_document_grading,
        {
            "generate": "generator",
            "rewrite": "query_rewriter",
            "web_search": "web_search",
        },
    )

    # Self-Correction Loop Edge: Rewriter -> Retriever
    workflow.add_edge("query_rewriter", "retriever")

    # Post-Search Edge: Web Search -> Generator
    workflow.add_edge("web_search", "generator")

    # Post-Generation Verification
    workflow.add_edge("generator", "hallucination_check")
    workflow.add_conditional_edges(
        "hallucination_check",
        route_after_hallucination_check,
        {"accept": END, "reject": "safe_abstention"},
    )
    workflow.add_edge("safe_abstention", END)

    return workflow.compile()

# Global compiled application
rag_app = create_rag_graph()
