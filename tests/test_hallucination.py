import pytest

from src.graph.nodes.hallucination import (
    hallucination_check_node,
    verify_citations_fast,
)
from src.graph.state import GraphState, DocumentChunk


def _retrieved_doc():
    return DocumentChunk(
        content="FastAPI dependency injection enables code reuse and request-scoped dependencies.",
        metadata={
            "title": "FastAPI Guide",
            "source": "fastapi.md",
            "header_path": "Dependencies > Dependency Injection",
        },
    )


def test_verify_citations_fast_accepts_matching_source():
    docs = [_retrieved_doc()]
    answer = "FastAPI uses Depends [Source: FastAPI Guide > Dependency Injection]."
    assert verify_citations_fast(answer, docs) is True


def test_verify_citations_fast_rejects_fabricated_source():
    docs = [_retrieved_doc()]
    answer = "Use fictional_method() [Source: Fabricated Manual > API Reference]."
    assert verify_citations_fast(answer, docs) is False


def test_verify_citations_fast_rejects_missing_citation():
    assert verify_citations_fast("FastAPI uses Depends.", [_retrieved_doc()]) is False


def test_verify_citations_fast_rejects_empty_evidence():
    assert verify_citations_fast("An answer [Source: FastAPI Guide].", []) is False


def test_hallucination_node_fails_closed_when_verifier_unavailable(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("src.core.llm.get_llm", unavailable)
    state: GraphState = {
        "generation": "FastAPI uses Depends [Source: FastAPI Guide > Dependency Injection].",
        "relevant_documents": [_retrieved_doc()],
        "execution_steps": [],
    }

    result = hallucination_check_node(state)

    assert result["hallucination_grade"] == "not grounded"
    assert "unavailable" in result["verification_reason"].lower()
    assert "hallucination_check" in result["execution_steps"]
