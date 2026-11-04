import pytest
from src.graph.workflow import create_rag_graph
from src.graph.state import GraphState, DocumentChunk

def test_workflow_compilation():
    graph = create_rag_graph()
    assert graph is not None
    # Verify graph has the defined node names
    assert "query_analysis" in graph.nodes
    assert "retriever" in graph.nodes
    assert "document_grader" in graph.nodes
    assert "query_rewriter" in graph.nodes
    assert "web_search" in graph.nodes
    assert "generator" in graph.nodes
    assert "hallucination_check" in graph.nodes

def test_workflow_controlled_execution(monkeypatch):
    """
    Simulates graph execution with deterministic node overrides to verify state propagation and edge traversal.
    """
    def mock_retriever(state: GraphState):
        steps = state.get("execution_steps", []).copy()
        steps.append("retriever")
        return {
            "documents": [
                DocumentChunk(
                    content="FastAPI dependency injection enables yield cleanup.",
                    metadata={"source": "fastapi.md", "title": "FastAPI Guide", "header_path": "Dependencies"}
                )
            ],
            "execution_steps": steps,
        }

    def mock_document_grader(state: GraphState):
        steps = state.get("execution_steps", []).copy()
        steps.append("document_grader")
        return {
            "relevant_documents": state.get("documents", []),
            "execution_steps": steps,
        }

    def mock_generator(state: GraphState):
        steps = state.get("execution_steps", []).copy()
        steps.append("generator")
        return {
            "generation": "Use yield in FastAPI for teardown. [Source: FastAPI Guide > Dependencies]",
            "sources": [{"source": "fastapi.md", "title": "FastAPI Guide", "header_path": "Dependencies"}],
            "execution_steps": steps,
        }

    monkeypatch.setattr("src.graph.workflow.retriever_node", mock_retriever)
    monkeypatch.setattr("src.graph.workflow.document_grader_node", mock_document_grader)
    monkeypatch.setattr("src.graph.workflow.generator_node", mock_generator)

    def mock_hallucination_check(state: GraphState):
        steps = state.get("execution_steps", []).copy()
        steps.append("hallucination_check")
        return {
            "hallucination_grade": "grounded",
            "verification_reason": "Deterministic test verifier approved the fixture.",
            "execution_steps": steps,
        }

    monkeypatch.setattr("src.graph.workflow.hallucination_check_node", mock_hallucination_check)

    app = create_rag_graph()
    initial_state: GraphState = {
        "question": "How to use yield in FastAPI?",
        "current_query": "How to use yield in FastAPI?",
        "retry_count": 0,
        "execution_steps": [],
    }

    final_state = app.invoke(initial_state)

    assert "generation" in final_state
    assert "Use yield in FastAPI" in final_state["generation"]
    assert len(final_state["sources"]) == 1
    assert final_state["hallucination_grade"] == "grounded"
    assert "query_analysis" in final_state["execution_steps"]
    assert "retriever" in final_state["execution_steps"]
    assert "document_grader" in final_state["execution_steps"]
    assert "generator" in final_state["execution_steps"]
    assert "hallucination_check" in final_state["execution_steps"]


def test_unverified_answer_is_routed_to_safe_abstention():
    from src.graph.edges import route_after_hallucination_check
    from src.graph.nodes.safe_abstention import safe_abstention_node

    state: GraphState = {
        "generation": "unverified draft",
        "sources": [{"source": "fabricated"}],
        "hallucination_grade": "not grounded",
        "execution_steps": ["hallucination_check"],
    }

    assert route_after_hallucination_check(state) == "reject"
    result = safe_abstention_node(state)
    assert result["hallucination_grade"] == "not grounded"
    assert result["sources"] == []
    assert "safe_abstention" in result["execution_steps"]
    assert "unverified draft" not in result["generation"]
