from typing import List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from src.graph.state import GraphState, DocumentChunk
from src.core.llm import get_llm

class ChunkGrade(BaseModel):
    chunk_index: int = Field(description="The index of the document chunk being evaluated (0-based)")
    is_relevant: bool = Field(description="True if relevant to the technical question, False if irrelevant")
    reasoning: str = Field(default="", description="Brief reason for the relevance score")

class BatchGraderOutput(BaseModel):
    evaluations: List[ChunkGrade] = Field(description="List of evaluations for each chunk")

BATCH_GRADER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are an expert, objective document relevance evaluator for technical documentation RAG pipelines.
Evaluate the provided numbered candidate document chunks against the user's technical question.
Grade each chunk as relevant (is_relevant=True) if it contains code, conceptual definitions, methods, or details that help answer the question.
Grade as irrelevant (is_relevant=False) only if the chunk is completely unrelated or off-topic.
Be generous toward technical context that can assist the answer."""
    ),
    (
        "human",
        """User Question:
{question}

Candidate Document Chunks:
{chunks_formatted}

Evaluate all chunks and return the relevance list."""
    )
])

def document_grader_node(state: GraphState) -> dict:
    """
    Evaluates retrieved chunks in a single batch LLM call for minimal latency.
    Retains only relevant documents in state['relevant_documents'].
    """
    question = state.get("question", "")
    documents = state.get("documents", [])
    steps = state.get("execution_steps", []).copy()
    steps.append("document_grader")

    if not documents:
        print("[Document Grader Node] No documents to grade.")
        return {
            "relevant_documents": [],
            "execution_steps": steps,
        }

    # Format all candidate chunks together for a single batch evaluation call
    chunks_text_list = []
    for idx, doc in enumerate(documents):
        title = doc.get("metadata", {}).get("title", "Document")
        header = doc.get("metadata", {}).get("header_path", "")
        preview = doc.get("content", "")[:600]
        chunks_text_list.append(f"[Chunk {idx}] (Title: {title} | Section: {header}):\n{preview}")

    chunks_formatted = "\n\n---\n\n".join(chunks_text_list)

    relevant_docs: List[DocumentChunk] = []

    try:
        llm = get_llm(temperature=0.0, with_failover=False)
        structured_grader = llm.with_structured_output(BatchGraderOutput)
        grader_chain = BATCH_GRADER_PROMPT | structured_grader

        # Single LLM call for all chunks!
        res: BatchGraderOutput = grader_chain.invoke({
            "question": question,
            "chunks_formatted": chunks_formatted
        })

        raw_indices = [item.chunk_index for item in res.evaluations]
        is_1_based = (
            bool(raw_indices)
            and min(raw_indices) >= 1
            and max(raw_indices) >= len(documents)
        )

        relevant_indices = set()
        for item in res.evaluations:
            if item.is_relevant:
                idx = item.chunk_index - 1 if is_1_based else item.chunk_index
                if 0 <= idx < len(documents):
                    relevant_indices.add(idx)

        # Resilient technical keyword preservation heuristic
        import re
        q_tokens = set(re.findall(r"@?[a-zA-Z_][a-zA-Z0-9_]{3,}", question.lower()))
        common_words = {"what", "when", "where", "which", "while", "with", "about", "using", "from", "have", "this", "that", "does", "explain", "implement"}
        key_tokens = [t for t in q_tokens if t not in common_words and len(t) >= 4]

        for idx, doc in enumerate(documents):
            content_lower = doc.get("content", "").lower()
            header_lower = doc.get("metadata", {}).get("header_path", "").lower()
            if any(tok in content_lower or tok in header_lower for tok in key_tokens if tok.startswith("@") or "_" in tok or len(tok) >= 6):
                relevant_indices.add(idx)

        # If relevant indices found, retain them
        if relevant_indices:
            for idx, doc in enumerate(documents):
                score = "relevant" if idx in relevant_indices else "irrelevant"
                graded_chunk: DocumentChunk = {
                    "content": doc.get("content", ""),
                    "metadata": doc.get("metadata", {}),
                    "relevance_score": score,
                    "reasoning": "Evaluated in batch"
                }
                if score == "relevant":
                    relevant_docs.append(graded_chunk)
        else:
            # All chunks evaluated as irrelevant -> empty list triggers self-corrective rewrite or web fallback
            relevant_docs = []

    except Exception as e:
        print(f"[Document Grader] Batch grading fallback due to: {e}")
        # Fast fallback: pass top documents through
        relevant_docs = documents

    print(f"[Document Grader Node] {len(relevant_docs)} / {len(documents)} chunks deemed relevant.")

    return {
        "relevant_documents": relevant_docs,
        "execution_steps": steps,
    }
