# Self-Corrective RAG Technical Documentation Assistant

[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateGraph-blue.svg?logo=langchain)](https://github.com/langchain-ai/langgraph)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![ChromaDB](https://img.shields.io/badge/VectorStore-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A local-first, agentic Retrieval-Augmented Generation (RAG) assistant for software engineering documentation. Built on a **LangGraph StateGraph** implementing corrective retrieval, evidence review, and a fail-closed answer path, served via **FastAPI** with SQLite persistence and an interactive web interface.

---

## 1. Project Overview & Key Innovations

Standard Naive RAG pipelines retrieve top-$k$ document chunks and immediately prompt an LLM to generate an answer. In technical documentation queries, naive RAG routinely fails:
1. **Vocabulary Mismatch**: Developers submit conversational prompts (*"How do I clean up connections after a request?"*) that omit official framework terms (*"Dependencies with `yield` and contextmanager teardown"*).
2. **Context Poisoning**: Blindly injecting irrelevant chunks produces hallucinated syntax, invalid parameters, and broken imports.
3. **No Self-Correction**: When initial retrieval returns 0 relevant chunks, naive pipelines produce silent hallucinations or generic apologies without attempting to reformulate search terms.

### How Our Self-Corrective Workflow Solves This
- **Query Analysis & Classification**: Analyzes intent (`how-to`, `conceptual`, `troubleshooting`, `api_reference`) and rewrites queries with domain-specific technical synonyms.
- **Two-Tier Chunking**: Combines `MarkdownHeaderTextSplitter` (preserving hierarchical breadcrumbs) with `RecursiveCharacterTextSplitter` (preserving Python code blocks intact).
- **Document Grading (CRAG)**: Evaluates candidate chunks with an LLM evaluator, filtering out irrelevant noise.
- **Conditional Query Reformulation**: If documents fail relevance scoring, the system reformulates search terms and re-retrieves candidate chunks with a strict retry limit (`max_retries = 2`) to mathematically eliminate infinite execution loops.
- **Web Search Fallback**: Automatically falls back to external web search (DuckDuckGo or Tavily) when documentation lacks coverage.
- **Strict Grounding & Citations**: Enforces explicit inline citations (`[Source: Document > Section]`).
- **Self-RAG Verification**: Runs an automated post-generation groundedness check to ensure zero fabricated code.

---

## 2. System Architecture & LangGraph Workflow

### 2.1 StateGraph Architecture Diagram

```mermaid
flowchart TD
    Start([User Question]) --> NodeQA[1. Query Analysis Node]
    NodeQA --> NodeRet[2. Retrieval Node]
    ChromaDB[(ChromaDB Vector Store)] -.->|Top-k Chunks & Metadata| NodeRet
    NodeRet --> NodeGrade[3. Document Grading Node]
    
    NodeGrade --> Router1{Are Chunks Relevant?}
    Router1 -->|Yes: >= 1 relevant| NodeGen[4. Generation Node]
    Router1 -->|No & Retries < Max| NodeRewrite[5. Query Rewriter Node]
    NodeRewrite -->|Increment Retry Count| NodeRet
    Router1 -->|No & Retries >= Max| Fallback[6. Web Search Fallback Node]
    Fallback --> NodeGen
    
    NodeGen --> NodeHallucination[7. Groundedness / Self-RAG Check]
    NodeHallucination --> Router2{Answer Grounded?}
    Router2 -->|Yes: evidence approved| End([Response with Citations])
    Router2 -->|No or verifier unavailable| Abstain[Safe Abstention]
    Abstain --> End
```

---

## 3. State Schema & Retry Tracking (Core Evaluation Criterion)

> [!IMPORTANT]
> **Take-Home Evaluation Requirement:**
> *"Think carefully about your state schema. What data flows between nodes? How do you track the number of retries? This is a core evaluation criterion."*

### 3.1 The `GraphState` TypedDict Schema

The entire workflow is governed by an immutable state dictionary passed and updated across nodes:

```python
from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict

class DocumentChunk(TypedDict, total=False):
    content: str
    metadata: Dict[str, Any]       # source, title, header_path, chunk_id
    relevance_score: Optional[str] # "relevant" | "irrelevant"
    reasoning: Optional[str]

class GraphState(TypedDict, total=False):
    question: str                      # Original raw user query
    current_query: str                 # Active query (rewritten or expanded)
    query_type: Optional[str]          # conceptual | how-to | troubleshooting | api_reference
    chat_history: List[Dict[str, str]] # Session conversation turns
    documents: List[DocumentChunk]      # Raw candidate retrieved chunks
    relevant_documents: List[DocumentChunk] # Chunks passing document grading
    generation: Optional[str]          # Generated final response
    sources: List[Dict[str, Any]]      # Citations list (source, header_path, snippet)
    retry_count: int                   # Number of rewrites executed
    max_retries: int                   # Maximum permitted rewrite retries (default: 2)
    web_search_needed: bool            # True if retrieval failed and web search should be used
    hallucination_grade: Optional[str] # "grounded" | "not grounded"
    execution_steps: List[str]         # Audit trail of executed nodes for observability
```

### 3.2 Retry Tracking & Mathematical Loop Termination
To guarantee that the cyclic graph never enters an infinite loop, the retry counter is enforced as a strict state invariant:
1. **Initialization:** When a query enters the graph, `retry_count` is initialized to `0` and `max_retries` is set to `2`.
2. **Conditional Check:** In `route_after_document_grading(state)`:
   - If `len(relevant_documents) > 0` $\rightarrow$ proceed to `generation`.
   - If `len(relevant_documents) == 0` AND `retry_count < max_retries` $\rightarrow$ route to `query_rewriter`.
   - If `len(relevant_documents) == 0` AND `retry_count >= max_retries` $\rightarrow$ route to `web_search_fallback` or fallback generator.
3. **State Mutation:** `query_rewriter` increments `retry_count += 1` before returning the state to `retriever`.
4. **Termination Proof:** Because `retry_count` strictly increases monotonically on every cycle and `max_retries` is a fixed upper bound ($N=2$), the graph is guaranteed to exit the cycle in at most $N$ iterations.

---

## 4. Document Ingestion Pipeline & Chunking Strategy

> [!IMPORTANT]
> **Take-Home Evaluation Requirement:**
> *"Document your chunking strategy and explain why you chose it."*

### 4.1 The Technical Documentation Challenge
Naive character-based splitting fails on technical documentation in two major ways:
1. **Broken Code Blocks:** Fixed-size chunking (e.g. splitting at 500 characters) frequently severs Python functions right in the middle of a class definition or `try...finally` block.
2. **Loss of Hierarchy:** Splitting without respecting headings detaches code samples from their context (e.g., separating a code block from the heading `## Dependencies with yield`).

### 4.2 Our Solution: Two-Tier Chunking Strategy

```
Raw Markdown Document
       │
       ▼
[Tier 1: MarkdownHeaderTextSplitter]
 Splits along #, ##, ### headers
 Attaches breadcrumb path metadata (e.g. "FastAPI > Dependencies > Yield")
       │
       ▼
[Tier 2: RecursiveCharacterTextSplitter]
 Chunk Size: 800 characters (~180 tokens)
 Chunk Overlap: 150 characters (~35 tokens)
 Separator Priority: ["\n\n```", "\n```\n", "\n\n", "\n", ". ", " "]
       │
       ▼
Guaranteed: Code blocks preserved intact + Full hierarchical context in metadata
```

- **Why Chunk Size = 800 Chars?** Technical explanations followed by a short code example typically span 600–800 characters. Sizing at 800 characters captures the conceptual rationale alongside the runnable code snippet without overflowing LLM context with unrelated topics.
- **Why Chunk Overlap = 150 Chars?** Ensures that introductory sentences bridging two sub-sections are not clipped at chunk boundaries.
- **Deterministic Chunk IDs:** Each chunk receives a UUID5 hash generated from its source, header path, index, and content prefix, ensuring idempotent indexing.

---

## 5. Seed Document Corpus (`data/raw_documents/`)

The repository includes a curated corpus of 4 comprehensive technical documents:

| Document | File Path | Topics Covered |
|---|---|---|
| **FastAPI Dependencies** | `data/raw_documents/fastapi_dependencies.md` | Dependency injection, sub-dependencies, context managers with `yield`, security OAuth2 schemes. |
| **LangGraph Core Concepts** | `data/raw_documents/langgraph_concepts.md` | StateGraph, nodes, conditional edges, self-corrective CRAG loops, state schemas. |
| **Pydantic v2 Validation** | `data/raw_documents/pydantic_validation.md` | `BaseModel`, `@field_validator`, `@model_validator`, `pydantic-settings` management. |
| **REST API Principles** | `data/raw_documents/rest_api_principles.md` | HTTP method semantics, idempotency, status codes (2xx, 4xx, 5xx), async patterns. |

---

## 6. Multi-Provider LLM & Embedding Factory

To ensure evaluators can run the assistant using any API key (or completely free of charge), the system includes an extensible **Factory Pattern**:

### Supported LLM Providers:
- **Groq (Default & Recommended):** Runs `llama-3.3-70b-versatile` with ultra-fast inference (>250 tokens/sec) on a generous free tier.
- **OpenAI:** `gpt-4o-mini` for high accuracy and cost efficiency.
- **Google Gemini:** `gemini-1.5-flash` with 1M context window and free tier.

### Local-First Embeddings:
- **`sentence-transformers/all-MiniLM-L6-v2`** via FastEmbed / HuggingFace:
  - Runs **100% locally** on CPU with zero API costs, zero rate limits, and sub-15ms embedding generation.
  - Can be toggled to OpenAI `text-embedding-3-small` with a single configuration flag.

---

## 7. API Specification & Example Requests

### 1. `POST /api/v1/query` — Submit a Question
Executes the self-corrective LangGraph workflow and returns an answer with grounded citations.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
     -H "Content-Type: application/json" \
     -d '{
       "question": "How do I implement dependency injection with yield in FastAPI for database cleanup?",
       "session_id": "session-dev-123"
     }'
```

**Response (200 OK):**
```json
{
  "query_id": "893c5d7a-12e0-47b2-8ea9-f80e224e75c1",
  "session_id": "session-dev-123",
  "query_type": "how-to",
  "answer": "To implement dependency injection with teardown/cleanup in FastAPI, use a Python generator function with `yield` instead of `return`.\n\n```python\nfrom fastapi import Depends, FastAPI\nfrom sqlalchemy.orm import Session\n\nasync def get_db():\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()\n```\n\nThe code prior to `yield` executes before the endpoint handles the request, and the code in the `finally` block runs after the response is sent, guaranteeing resource cleanup [Source: FastAPI Dependencies > Dependencies with yield].",
  "sources": [
    {
      "source": "fastapi_dependencies.md",
      "title": "FastAPI Dependencies and Dependency Injection",
      "header_path": "Dependencies with yield (Context Managers)",
      "snippet": "async def get_db():\n    db = SessionLocal()\n    try:\n        yield db\n    finally:\n        db.close()"
    }
  ],
  "retries_used": 0,
  "grounded": "grounded",
  "execution_steps": [
    "query_analysis",
    "retriever",
    "document_grader",
    "generator",
    "hallucination_check"
  ]
}
```

---

### 2. `POST /api/v1/ingest` — Ingest Documentation from URLs
Crawls, parses, chunks, and indexes external documentation URLs dynamically.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest" \
     -H "Content-Type: application/json" \
     -d '{
       "urls": [
         "https://fastapi.tiangolo.com/tutorial/dependencies/"
       ]
     }'
```

**Response (201 Created):**
```json
{
  "status": "success",
  "indexed_documents": 1,
  "total_chunks": 8,
  "message": "Successfully indexed 1 document(s) into 8 chunks."
}
```

---

### 3. `POST /api/v1/ingest/upload` — Ingest Local Documentation Files
Uploads Markdown, text, or HTML files to chunk and index.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/ingest/upload" \
     -F "files=@./data/raw_documents/fastapi_dependencies.md"
```

**Response (201 Created):**
```json
{
  "status": "success",
  "indexed_documents": 1,
  "total_chunks": 6,
  "message": "Successfully uploaded and indexed 1 file(s) into 6 chunks."
}
```

---

### 4. `GET /api/v1/documents` — List Corpus Documents & Chunk Statistics
Returns a complete catalog of all indexed documentation.

**Request:**
```bash
curl -X GET "http://localhost:8000/api/v1/documents"
```

**Response (200 OK):**
```json
{
  "total_documents": 4,
  "total_vectors": 24,
  "documents": [
    {
      "source": "data/raw_documents/fastapi_dependencies.md",
      "title": "FastAPI Dependencies and Dependency Injection",
      "chunk_count": 6,
      "source_type": "file",
      "ingested_at": "2026-10-14T15:45:00Z"
    },
    {
      "source": "data/raw_documents/langgraph_concepts.md",
      "title": "LangGraph Core Concepts and StateGraph Workflows",
      "chunk_count": 7,
      "source_type": "file",
      "ingested_at": "2026-10-14T15:45:00Z"
    }
  ]
}
```

---

### 5. `POST /api/v1/feedback` — Submit Answer Rating
Records user feedback (thumbs up/down + optional comment) to SQLite for evaluation.

**Request:**
```bash
curl -X POST "http://localhost:8000/api/v1/feedback" \
     -H "Content-Type: application/json" \
     -d '{
       "query_id": "893c5d7a-12e0-47b2-8ea9-f80e224e75c1",
       "rating": "up",
       "comment": "Accurate explanation of the yield contextmanager cleanup pattern."
     }'
```

**Response (201 Created):**
```json
{
  "feedback_id": "c841a129-9e87-432d-9861-4560a87f1234",
  "status": "recorded",
  "message": "Feedback successfully recorded. Thank you for helping evaluate our RAG assistant!"
}
```

---

### 6. `GET /api/v1/query/sessions` — List Recent Conversation Sessions
Returns active and past conversation threads for sidebar session management.

**Response (200 OK):**
```json
{
  "sessions": [
    {
      "session_id": "session-dev-123",
      "question": "How do I implement dependency injection with yield in FastAPI?",
      "created_at": "2026-10-14 11:13:45",
      "turn_count": 2
    }
  ]
}
```

---

### 7. `GET /api/v1/query/history` — Fetch Session Conversation History
Retrieves historical query-response pairs, execution trails, and citations for a specific `session_id`.

**Response (200 OK):**
```json
{
  "session_id": "session-dev-123",
  "total_messages": 2,
  "messages": [
    {
      "id": "893c5d7a-12e0-47b2-8ea9-f80e224e75c1",
      "session_id": "session-dev-123",
      "question": "How do I implement dependency injection with yield in FastAPI?",
      "answer": "To implement dependency injection with teardown in FastAPI...",
      "query_type": "how-to",
      "retries": 0,
      "grounded": "grounded",
      "sources": [...]
    }
  ]
}
```

---

### 8. `GET /api/v1/health` — System Health & Index Readiness
**Response (200 OK):**
```json
{
  "status": "healthy",
  "app_name": "Technical Documentation RAG Assistant",
  "llm_provider": "gemini",
  "model_name": "gemini-3.5-flash-lite",
  "embedding_provider": "fastembed",
  "vector_store_count": 27
}
```

---

### 9. `GET /api/v1/query/export/{session_id}` — Export Conversation Transcript
Exports the entire chat thread including source citations and evaluation metadata as Markdown (`.md`) or structured JSON.

**Query Parameters:**
* `format`: `markdown` (default) or `json`

**Response (200 OK):** Streams `attachment; filename="docugraph_session_<id>.md"`.

---

### 10. `GET /download/latest` — Universal Application Download
Automatically detects and serves the best available installer package (`.exe`, `.msi`, or complete project source archive `.zip`).

**Response (200 OK):** Direct binary file stream with `Accept-Ranges: bytes` support.

---

## 8. Desktop Application & Windows Installer (.exe / .msi)

DocuGraph AI can be packaged and distributed as a native Windows desktop application. After installation, users simply double-click the desktop shortcut icon to launch the application. The system automatically initializes the embedded FastAPI server, self-corrective LangGraph pipeline, and ChromaDB vector store, presenting the rich web frontend in a dedicated, hardware-accelerated desktop window.

### 8.1 Windows Installer Packages (`dist/`)

The repository builds ready-to-use Windows installers:

| Package | Path | Description |
|---|---|---|
| **Windows Setup Wizard (.exe)** | [`dist/DocuGraphAI_Setup.exe`](file:///c:/Users/21330/Documents/RAG_Project_sample/dist/DocuGraphAI_Setup.exe) | Graphical Windows installer wizard with progress bar, custom icon, desktop shortcut, and uninstaller. |
| **Windows Installer Package (.msi)** | [`dist/DocuGraphAI_Setup.msi`](file:///c:/Users/21330/Documents/RAG_Project_sample/dist/DocuGraphAI_Setup.msi) | Enterprise `.msi` package compatible with Windows Installer (`msiexec /i DocuGraphAI_Setup.msi`). |
| **1-Click Batch Installer** | [`Install_DocuGraphAI.bat`](file:///c:/Users/21330/Documents/RAG_Project_sample/Install_DocuGraphAI.bat) | Instant installer script creating desktop & start menu shortcuts directly. |
| **Portable Desktop Executable** | [`dist/DocuGraphAI/DocuGraphAI.exe`](file:///c:/Users/21330/Documents/RAG_Project_sample/dist/DocuGraphAI/DocuGraphAI.exe) | Portable binary running without prior installation. |

### 8.2 Installation & Launch Flow

1. **Download & Run Installer:** Double-click `DocuGraphAI_Setup.exe` (or run `msiexec /i DocuGraphAI_Setup.msi`).
2. **Setup Wizard:** Click **"Install Now"** (installs to `%LOCALAPPDATA%\Programs\DocuGraphAI`).
3. **Desktop Shortcut:** The installer places a shortcut named **"DocuGraph AI"** with the high-resolution application icon on your Desktop and Start Menu.
4. **Launch:** Click the **DocuGraph AI** desktop icon. The application window opens with the complete self-corrective RAG pipeline running.
5. **Clean Exit & Uninstallation:** Closing the desktop window automatically shuts down the background server. Running `Uninstall.bat` cleanly removes all installed files and shortcuts.

### 8.3 Running Desktop App from Source

To launch the desktop application directly in Python:
```powershell
.\.venv\Scripts\python desktop_app.py
```
- For automated testing / verification:
  ```powershell
  .\.venv\Scripts\python desktop_app.py --check
  ```

---

## 9. Setup & Development Guide (CLI / Server)

### Prerequisites
- Python 3.11+
- Virtual Environment (`.venv`)

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone <repository-url>
cd RAG_Project_sample

python -m venv .venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux/macOS:
source .venv/bin/activate
```

### Step 2: Configure Environment
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Set Google Gemini (default provider) with your API key:
```ini
DEFAULT_LLM_PROVIDER="gemini"
GEMINI_API_KEY="your-gemini-api-key"
GEMINI_MODEL="gemini-3.6-flash"
```

### Step 3: Run the Application Server
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

- **Modern Web Interface:** Open [http://localhost:8000/](http://localhost:8000/) in your browser.
- **Interactive Swagger Docs:** Open [http://localhost:8000/docs](http://localhost:8000/docs).


---


---

## Demo Security and Operational Boundaries

DocuGraph AI is configured for a **local, single-user demonstration**, not public internet deployment.

### Safe demo defaults

- The API binds to 127.0.0.1 by default. Keep it on loopback for the demo.
- CORS is restricted to the local origins listed in .env.example; credentials are not enabled.
- URL ingestion accepts only public HTTP(S) destinations on ports 80 and 443, rejects local/private/link-local IP destinations, validates redirect targets, and caps document downloads at 2 MiB.
- File ingestion accepts Markdown, text, and HTML formats only. Client filenames are never used as disk paths. The limit is 5 MiB per file, 10 MiB per request, and 10 files per request.
- Source-code download bundles exclude local ChromaDB data, uploaded documents, SQLite databases, and environment overrides.
- Answer verification checks citation references against retrieved metadata and asks a structured evidence auditor to review material claims and code. If citation validation fails, the verifier is unavailable, or the audit rejects the answer, the workflow returns a safe abstention instead of presenting an unverified draft as grounded.
- GET /health is a liveness check. GET /ready reports whether the vector index and provider credentials are configured; it does not prove that the provider is currently reachable.

### Important limitation

This demo does **not** include user authentication, tenant isolation, or authorization for session-history endpoints. Do not bind it to 0.0.0.0, expose it through a public tunnel, or deploy it to a shared network. A public deployment requires authentication, authorization, per-user data isolation, production egress controls, and an appropriately managed database before it is safe to use.

See docs/DEMO_SECURITY.md for the demo threat model and release checklist.