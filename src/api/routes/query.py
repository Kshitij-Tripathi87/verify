import time
import uuid
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, status

from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    SourceCitation,
    HistoryResponse,
    HistoryItem,
    SessionListResponse,
    SessionListItem,
)
from src.graph.workflow import rag_app
from src.storage.repository import storage_repo

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit question to documentation assistant",
    description="Executes the self-corrective LangGraph workflow and returns a grounded answer with citations."
)
async def query_documentation(request: QueryRequest) -> QueryResponse:
    if not request.question or not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or solely whitespace."
        )

    session_id = request.session_id or str(uuid.uuid4())
    start_time = time.perf_counter()
    try:
        # Load previous conversation history for multi-turn session memory
        past_turns = storage_repo.get_session_queries(session_id, limit=6)
        formatted_history = []
        for turn in past_turns:
            q_text = turn.get("question")
            a_text = turn.get("answer")
            if q_text:
                formatted_history.append({"role": "user", "content": q_text})
            if a_text:
                formatted_history.append({"role": "assistant", "content": a_text})

        initial_state = {
            "question": request.question,
            "current_query": request.question,
            "chat_history": formatted_history,
            "retry_count": 0,
            "execution_steps": [],
        }

        # Execute LangGraph StateGraph
        final_state = rag_app.invoke(initial_state)


        answer = final_state.get("generation", "No answer generated.")
        query_type = final_state.get("query_type", "conceptual")
        raw_sources = final_state.get("sources", [])
        retries = final_state.get("retry_count", 0)
        grounded = final_state.get("hallucination_grade", "grounded")
        steps = final_state.get("execution_steps", [])
        execution_time_ms = round((time.perf_counter() - start_time) * 1000, 2)

        citations = [
            SourceCitation(
                source=s.get("source", ""),
                title=s.get("title", ""),
                header_path=s.get("header_path", ""),
                snippet=s.get("snippet", ""),
            )
            for s in raw_sources
        ]

        # Record in SQLite audit log
        query_id = storage_repo.save_query(
            question=request.question,
            answer=answer,
            session_id=session_id,
            query_type=query_type,
            retries=retries,
            sources=raw_sources,
            grounded=grounded,
        )

        response = QueryResponse(
            query_id=query_id,
            session_id=session_id,
            answer=answer,
            query_type=query_type,
            sources=citations,
            retries_used=retries,
            grounded=grounded,
            execution_steps=steps,
            execution_time_ms=execution_time_ms,
        )

        return response


    except Exception as e:
        print(f"[Query API] Exception during graph execution ({e}). Activating high-availability extractive fallback.")
        try:
            from src.core.vector_store import vector_store_manager
            from src.core.fallback import synthesize_extractive_answer
            fallback_docs = vector_store_manager.similarity_search(query=request.question, k=3)
            formatted_fallback_docs = [
                {"content": d.page_content, "metadata": d.metadata}
                for d in fallback_docs
            ]
            fallback_answer = synthesize_extractive_answer(
                request.question,
                formatted_fallback_docs,
                error_reason=str(e),
            )
            fallback_citations = [
                SourceCitation(
                    source=d.metadata.get("source", "corpus"),
                    title=d.metadata.get("title", "Technical Document"),
                    header_path=d.metadata.get("header_path", "General"),
                    snippet=d.page_content[:200] + "...",
                )
                for d in fallback_docs
            ]
            query_id = storage_repo.save_query(
                question=request.question,
                answer=fallback_answer,
                session_id=session_id,
                query_type="conceptual",
                retries=0,
                sources=[c.model_dump() for c in fallback_citations],
                grounded="grounded",
            )
            return QueryResponse(
                query_id=query_id,
                session_id=session_id,
                answer=fallback_answer,
                query_type="conceptual",
                sources=fallback_citations,
                retries_used=0,
                grounded="grounded",
                execution_steps=["retriever", "extractive_synthesizer_fallback"],
                execution_time_ms=round((time.perf_counter() - start_time) * 1000, 2),
            )
        except Exception as inner_e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error executing documentation assistant: {str(e)} (Fallback error: {str(inner_e)})"
            )


@router.get(
    "/history",
    response_model=HistoryResponse,
    summary="Retrieve session query history",
    description="Returns previous queries and answers for a given session ID."
)
async def get_query_history(
    session_id: str = Query(..., description="Unique session identifier"),
    limit: int = Query(50, ge=1, le=100, description="Max messages to fetch")
) -> HistoryResponse:
    try:
        records = storage_repo.get_session_queries(session_id=session_id, limit=limit)
        items = []
        for r in records:
            sources = [
                SourceCitation(
                    source=s.get("source", ""),
                    title=s.get("title", ""),
                    header_path=s.get("header_path", ""),
                    snippet=s.get("snippet", ""),
                )
                for s in r.get("sources", [])
            ]
            items.append(
                HistoryItem(
                    id=r["id"],
                    session_id=r["session_id"],
                    question=r["question"],
                    answer=r["answer"],
                    query_type=r.get("query_type", "conceptual"),
                    retries=r.get("retries", 0),
                    grounded=r.get("grounded", "grounded"),
                    sources=sources,
                    created_at=r.get("created_at"),
                )
            )

        return HistoryResponse(
            session_id=session_id,
            total_messages=len(items),
            messages=items,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch session history: {str(e)}"
        )


@router.get(
    "/sessions",
    response_model=SessionListResponse,
    summary="List recent active conversation sessions",
    description="Returns recent conversation sessions for sidebar navigation."
)
async def list_recent_sessions(
    limit: int = Query(20, ge=1, le=50, description="Max sessions to return")
) -> SessionListResponse:
    try:
        rows = storage_repo.get_recent_sessions(limit=limit)
        sessions = [
            SessionListItem(
                session_id=row["session_id"],
                question=row["question"],
                created_at=str(row["created_at"]),
                turn_count=row["turn_count"],
            )
            for row in rows
        ]
        return SessionListResponse(sessions=sessions)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch sessions: {str(e)}"
        )


@router.delete(
    "/history/{session_id}",
    summary="Clear conversation history",
    description="Deletes all queries associated with a specific session."
)
async def clear_session_history(session_id: str):
    try:
        deleted = storage_repo.clear_session_queries(session_id)
        return {"session_id": session_id, "deleted_count": deleted, "status": "cleared"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear session: {str(e)}"
        )


@router.get(
    "/export/{session_id}",
    summary="Export conversation session as Markdown or JSON",
    description="Generates an export of the conversation session including sources and telemetry."
)
async def export_session(
    session_id: str,
    format: str = Query("markdown", description="Format to export ('markdown' or 'json')")
):
    history = storage_repo.get_session_history(session_id, limit=100)
    if not history:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No conversation history found for session ID: {session_id}"
        )

    if format.lower() == "json":
        return {
            "session_id": session_id,
            "export_time": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            "messages": history,
        }

    # Markdown format export
    lines = [
        f"# DocuGraph AI Conversation Export",
        f"- **Session ID:** `{session_id}`",
        f"- **Exported At:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        f"- **Total Turns:** {len(history)}",
        f"\n---\n"
    ]
    for idx, item in enumerate(history, 1):
        q = item.get("question", "")
        a = item.get("answer", "")
        grounded = item.get("grounded", "grounded")
        retries = item.get("retries", 0)
        lines.append(f"### Q{idx}: {q}")
        lines.append(f"*{grounded.title()} • {retries} query rewrite(s)*\n")
        lines.append(f"{a}\n")
        sources = item.get("sources", [])
        if sources:
            lines.append("**Sources:**")
            for s in sources:
                title = s.get("title", "Doc")
                sec = s.get("header_path", "")
                lines.append(f"- *{title}* (`{sec}`)")
        lines.append("\n---\n")

    md_content = "\n".join(lines)
    from fastapi.responses import Response
    return Response(
        content=md_content,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="docugraph_session_{session_id[:8]}.md"'}
    )
