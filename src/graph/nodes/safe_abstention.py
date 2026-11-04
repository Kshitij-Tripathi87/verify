from src.graph.state import GraphState


ABSTENTION_MESSAGE = (
    "I could not verify this answer against the retrieved documentation. "
    "To avoid presenting unsupported technical guidance, I have withheld the "
    "draft answer. Try narrowing the question or adding an authoritative source."
)


def safe_abstention_node(state: GraphState) -> dict:
    """Replace an unverified draft with a clear, non-technical abstention."""
    steps = state.get("execution_steps", []).copy()
    steps.append("safe_abstention")
    return {
        "generation": ABSTENTION_MESSAGE,
        "sources": [],
        "hallucination_grade": "not grounded",
        "execution_steps": steps,
    }
