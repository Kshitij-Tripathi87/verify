/**
 * DocuGraph AI - Enterprise Self-Corrective RAG Web Client
 * LangGraph Engine + ChromaDB + FastEmbed + Google Gemini
 */

// Global State
let currentSessionId = localStorage.getItem("docugraph_session_id") || generateSessionId();
localStorage.setItem("docugraph_session_id", currentSessionId);

let allLoadedDocuments = [];
let pendingFeedbackQueryId = null;

// Initialization on DOM Ready
document.addEventListener("DOMContentLoaded", () => {
  initTextareaAutoGrow();
  checkHealth();
  loadRecentSessions();
  loadDocuments();
  restoreSessionHistory(currentSessionId);
});

function generateSessionId() {
  return "sess_" + Math.random().toString(36).substring(2, 9) + "_" + Date.now().toString(36);
}

/* ==========================================================================
   Tab Navigation
   ========================================================================== */
function switchTab(tabName) {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.classList.remove("active");
    btn.setAttribute("aria-selected", "false");
  });
  document.querySelectorAll(".tab-pane").forEach((pane) => pane.classList.remove("active"));

  const targetBtn = document.getElementById(`tab-${tabName}-btn`);
  const targetPane = document.getElementById(`tab-${tabName}`);
  
  if (targetBtn) {
    targetBtn.classList.add("active");
    targetBtn.setAttribute("aria-selected", "true");
  }
  if (targetPane) {
    targetPane.classList.add("active");
  }

  if (tabName === "corpus") {
    loadDocuments();
  }
}

/* ==========================================================================
   Textarea Auto-grow & Keyboard Listeners
   ========================================================================== */
function initTextareaAutoGrow() {
  const textarea = document.getElementById("user-input");
  if (!textarea) return;

  textarea.addEventListener("input", () => {
    textarea.style.height = "auto";
    textarea.style.height = Math.min(textarea.scrollHeight, 140) + "px";
  });

  textarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
}

/* ==========================================================================
   Health & Engine Telemetry
   ========================================================================== */
async function checkHealth() {
  try {
    const res = await fetch("/api/v1/health");
    if (!res.ok) throw new Error("Health check degraded");
    const data = await res.json();

    const dot = document.getElementById("health-dot");
    const modelBadge = document.getElementById("model-badge");
    const vectorsBadge = document.getElementById("vectors-badge");
    const llmLabel = document.getElementById("current-llm-label");

    if (dot) dot.style.background = "#10b981";
    if (modelBadge) modelBadge.innerText = data.model_name || data.llm_provider.toUpperCase();
    if (vectorsBadge) vectorsBadge.innerText = `${data.vector_store_count} Chunks`;
    if (llmLabel) llmLabel.innerText = data.model_name || data.llm_provider;
  } catch (err) {
    const dot = document.getElementById("health-dot");
    const modelBadge = document.getElementById("model-badge");
    if (dot) dot.style.background = "#f43f5e";
    if (modelBadge) modelBadge.innerText = "Offline / Reconnecting";
  }
}

/* ==========================================================================
   Session Management & Sidebar
   ========================================================================== */
function startNewChat() {
  currentSessionId = generateSessionId();
  localStorage.setItem("docugraph_session_id", currentSessionId);

  const container = document.getElementById("chat-messages");
  container.innerHTML = getEmptyStateHtml();
  
  loadRecentSessions();
  showToast("Started new conversation session", "info");

  const input = document.getElementById("user-input");
  if (input) {
    input.value = "";
    input.style.height = "auto";
    input.focus();
  }
}

async function loadRecentSessions() {
  const listEl = document.getElementById("sessions-list");
  if (!listEl) return;

  try {
    const res = await fetch("/api/v1/query/sessions?limit=15");
    if (!res.ok) return;
    const data = await res.json();

    if (!data.sessions || data.sessions.length === 0) {
      listEl.innerHTML = `<div style="font-size: 0.75rem; color: var(--text-dim); padding: 0.5rem 0.25rem;">No previous sessions found</div>`;
      return;
    }

    listEl.innerHTML = data.sessions.map((s) => {
      const isActive = s.session_id === currentSessionId;
      const encodedSessionId = encodeURIComponent(String(s.session_id)).replace(/'/g, "%27");
      return `
        <button type="button" class="session-item ${isActive ? 'active' : ''}" data-session-id="${escapeHtml(encodedSessionId)}" title="${escapeHtml(s.question)}">
          <span class="session-title-text">${escapeHtml(s.question || 'Conversation')}</span>
          <span class="session-turns-pill">${Number(s.turn_count) || 0}</span>
        </button>
      `;
    }).join("");

    // Keep untrusted session identifiers out of inline JavaScript handlers.
    listEl.querySelectorAll("[data-session-id]").forEach((item) => {
      item.addEventListener("click", () => {
        try {
          selectSession(decodeURIComponent(item.dataset.sessionId || ""));
        } catch (err) {
          console.error("Invalid session identifier in session list", err);
        }
      });
    });
  } catch (err) {
    console.error("Failed to load sessions", err);
  }
}

async function selectSession(sessionId) {
  currentSessionId = sessionId;
  localStorage.setItem("docugraph_session_id", currentSessionId);
  await restoreSessionHistory(sessionId);
  loadRecentSessions();
}

async function restoreSessionHistory(sessionId) {
  try {
    const res = await fetch(`/api/v1/query/history?session_id=${encodeURIComponent(sessionId)}&limit=50`);
    if (!res.ok) return;
    const data = await res.json();

    if (!data.messages || data.messages.length === 0) {
      // Show empty state if no prior turns in this session
      const container = document.getElementById("chat-messages");
      container.innerHTML = getEmptyStateHtml();
      return;
    }

    // Render historical messages
    const container = document.getElementById("chat-messages");
    container.innerHTML = "";

    data.messages.forEach((msg) => {
      appendUserMessage(msg.question, false);
      appendAssistantMessage({
        query_id: msg.id,
        session_id: msg.session_id,
        answer: msg.answer,
        query_type: msg.query_type,
        retries_used: msg.retries,
        grounded: msg.grounded,
        sources: msg.sources,
        execution_steps: ["query_analysis", "retriever", "document_grader", "generator", "hallucination_check"],
        execution_time_ms: 0,
      }, false);
    });

    container.scrollTop = container.scrollHeight;
  } catch (err) {
    console.error("Failed to restore history", err);
  }
}

/* ==========================================================================
   Chat Send & Response Handling
   ========================================================================== */
function sendSuggestion(promptText) {
  const input = document.getElementById("user-input");
  if (!input) return;
  input.value = promptText;
  handleSend();
}

async function handleSend() {
  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-button");
  const query = input.value.trim();
  if (!query) return;

  // Clear empty state if present
  const emptyState = document.getElementById("empty-state");
  if (emptyState) emptyState.remove();

  // Append user bubble
  appendUserMessage(query, true);
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;

  // Append in-flight loading animation
  const loadingId = appendLoadingIndicator();

  try {
    const res = await fetch("/api/v1/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: query, session_id: currentSessionId })
    });

    removeMessage(loadingId);

    if (!res.ok) {
      const errorData = await res.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    appendAssistantMessage(data, true);
    checkHealth();
    loadRecentSessions();
  } catch (err) {
    removeMessage(loadingId);
    appendErrorMessage(err.message, query);
  } finally {
    sendBtn.disabled = false;
    input.focus();
  }
}

function retryQuery(queryText) {
  const input = document.getElementById("user-input");
  if (input) {
    input.value = queryText;
    handleSend();
  }
}

function appendUserMessage(text, scroll = true) {
  const container = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "message-row user-row";
  div.innerHTML = `
    <div class="message-body">
      <div class="user-bubble">${escapeHtml(text)}</div>
    </div>
    <div class="message-avatar user-avatar" title="You">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>
    </div>
  `;
  container.appendChild(div);
  if (scroll) container.scrollTop = container.scrollHeight;
}

function appendLoadingIndicator() {
  const container = document.getElementById("chat-messages");
  const id = "loading-" + Date.now();
  const div = document.createElement("div");
  div.id = id;
  div.className = "message-row";
  div.innerHTML = `
    <div class="message-avatar bot-avatar">⚡</div>
    <div class="message-body">
      <div class="bot-bubble" style="display: flex; align-items: center; gap: 0.85rem; padding: 1rem 1.25rem;">
        <span class="spin" style="display: inline-block;">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#818cf8" stroke-width="2.5"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
        </span>
        <div style="display: flex; flex-direction: column; gap: 0.15rem;">
          <span style="font-weight: 600; color: #f1f5f9; font-size: 0.9rem;">Executing LangGraph State Machine...</span>
          <span style="font-size: 0.75rem; color: var(--text-dim);">Query Analysis ➔ ChromaDB Retrieval ➔ Document Grader ➔ Failover-Protected Generator</span>
        </div>
      </div>
    </div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
  return id;
}

function removeMessage(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function appendErrorMessage(errText, failedQuery = "") {
  const container = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "message-row";
  const escapedQuery = failedQuery ? escapeHtml(failedQuery).replace(/'/g, "\\'") : "";
  const retryBtn = failedQuery ? `
    <button class="secondary-btn" style="margin-top: 0.75rem; font-size: 0.78rem; padding: 0.35rem 0.85rem; border-color: rgba(244, 63, 94, 0.4); color: #fda4af;" onclick="retryQuery('${escapedQuery}')">
      🔄 Retry Query
    </button>
  ` : "";

  div.innerHTML = `
    <div class="message-avatar bot-avatar" style="background: var(--accent-rose);">⚠️</div>
    <div class="message-body">
      <div class="bot-bubble" style="border-color: rgba(244, 63, 94, 0.4); background: rgba(244, 63, 94, 0.08);">
        <strong style="color: #fda4af;">Execution Notice:</strong>
        <p style="margin-top: 0.35rem; font-size: 0.88rem; color: #fecdd3;">${escapeHtml(errText)}</p>
        ${retryBtn}
      </div>
    </div>
  `;
  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

function appendAssistantMessage(data, scroll = true) {
  const container = document.getElementById("chat-messages");
  const div = document.createElement("div");
  div.className = "message-row";

  // Build LangGraph Traversal Nodes
  const steps = data.execution_steps && data.execution_steps.length > 0
    ? data.execution_steps
    : ["query_analysis", "retriever", "document_grader", "generator", "hallucination_check"];

  const stepsHtml = steps.map((step, idx) => {
    const isLast = idx === steps.length - 1;
    const cleanName = step.replace(/_/g, " ");
    return `
      <span class="trail-step ${isLast ? 'active' : ''}">
        ${isLast ? '✓ ' : ''}${cleanName}
      </span>
      ${!isLast ? '<span class="trail-arrow">➔</span>' : ''}
    `;
  }).join("");

  // Telemetry Pills
  const latencyHtml = data.execution_time_ms > 0
    ? `<span class="meta-pill latency">⏱️ ${(data.execution_time_ms / 1000).toFixed(2)}s</span>`
    : "";

  const retriesHtml = data.retries_used > 0
    ? `<span class="meta-pill retries">🔄 ${data.retries_used} rewrite${data.retries_used > 1 ? 's' : ''}</span>`
    : "";

  let groundedHtml = "";
  if (data.grounded === "extractive_offline_fallback" || (data.execution_steps && data.execution_steps.includes("extractive_synthesizer_fallback"))) {
    groundedHtml = `<span class="meta-pill" style="background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35);">🛡️ Rate-Limit Failover Activated</span>`;
  } else if (data.grounded === "grounded") {
    groundedHtml = `<span class="meta-pill grounded">✓ Self-RAG Grounded</span>`;
  } else {
    groundedHtml = `<span class="meta-pill disclaimer">⚠ External Knowledge</span>`;
  }

  // Citations Accordion
  let citationsHtml = "";
  if (data.sources && data.sources.length > 0) {
    const citationCards = data.sources.map((s, idx) => `
      <div class="citation-card">
        <div class="citation-top-row">
          <span class="citation-source-title">📄 ${escapeHtml(s.title || 'Technical Document')}</span>
          <span class="citation-header-path">${escapeHtml(s.header_path || 'Section')}</span>
        </div>
        <div class="citation-snippet-text">${escapeHtml(s.snippet || '')}</div>
      </div>
    `).join("");

    citationsHtml = `
      <div class="citations-section">
        <button class="citations-header-btn" onclick="toggleCitations(this)">
          <span>📚 Referenced Documentation Sources (${data.sources.length})</span>
          <span class="accordion-arrow">▼</span>
        </button>
        <div class="citation-cards-list">
          ${citationCards}
        </div>
      </div>
    `;
  }

  // Answer formatted with Markdown & Code Copying
  const formattedAnswer = renderMarkdownWithCode(data.answer);

  div.innerHTML = `
    <div class="message-avatar bot-avatar" title="DocuGraph Assistant">⚡</div>
    <div class="message-body" style="width: 100%;">
      <div class="bot-bubble">
        
        <!-- Interactive LangGraph Pipeline Trail -->
        <div class="pipeline-trail-box">
          <div class="trail-nodes">
            <span class="trail-title">Graph Trail:</span>
            ${stepsHtml}
          </div>
          <div class="trail-meta">
            ${latencyHtml}
            ${retriesHtml}
            ${groundedHtml}
          </div>
        </div>

        <!-- Rendered Answer Content -->
        <div class="markdown-body">
          ${formattedAnswer}
        </div>

        <!-- Collapsible Citations -->
        ${citationsHtml}

        <!-- Interactive Feedback & Copy Bar -->
        <div class="feedback-bar">
          <div class="feedback-buttons">
            <button class="fb-btn up" onclick="handleFeedback('${data.query_id}', 'up', this)" title="Accurate and helpful">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3zM7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg>
              Helpful
            </button>
            <button class="fb-btn down" onclick="handleFeedback('${data.query_id}', 'down', this)" title="Inaccurate or incomplete">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3zm7-13h3a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2h-3"/></svg>
              Inaccurate
            </button>
          </div>
          <button class="copy-answer-btn" onclick="copyFullAnswer(this)" title="Copy entire answer text">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            Copy Answer
          </button>
        </div>

      </div>
    </div>
  `;

  container.appendChild(div);
  if (scroll) container.scrollTop = container.scrollHeight;
}

function toggleCitations(btn) {
  const cardList = btn.nextElementSibling;
  const arrow = btn.querySelector(".accordion-arrow");
  if (cardList.style.display === "none") {
    cardList.style.display = "flex";
    if (arrow) arrow.innerText = "▼";
  } else {
    cardList.style.display = "none";
    if (arrow) arrow.innerText = "▶";
  }
}

/* ==========================================================================
   Feedback Submission
   ========================================================================== */
async function handleFeedback(queryId, rating, btn) {
  const parent = btn.parentElement;
  parent.querySelectorAll(".fb-btn").forEach((b) => b.classList.remove("active"));
  btn.classList.add("active");

  try {
    await fetch("/api/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query_id: queryId, rating: rating })
    });

    if (rating === "up") {
      showToast("Thank you for your feedback!", "success");
    } else {
      // Open qualitative feedback modal
      pendingFeedbackQueryId = queryId;
      openFeedbackModal();
    }
  } catch (err) {
    console.error("Feedback submission error", err);
  }
}

function openFeedbackModal() {
  const modal = document.getElementById("feedback-modal");
  if (modal) modal.style.display = "flex";
}

function closeFeedbackModal() {
  const modal = document.getElementById("feedback-modal");
  if (modal) modal.style.display = "none";
  const commentInput = document.getElementById("feedback-comment-input");
  if (commentInput) commentInput.value = "";
  pendingFeedbackQueryId = null;
}

async function submitDetailedFeedback() {
  const commentInput = document.getElementById("feedback-comment-input");
  const comment = commentInput ? commentInput.value.trim() : "";
  if (!pendingFeedbackQueryId || !comment) {
    closeFeedbackModal();
    return;
  }

  try {
    await fetch("/api/v1/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_id: pendingFeedbackQueryId,
        rating: "down",
        comment: comment
      })
    });
    showToast("Feedback recorded. We are continuously improving our answers!", "success");
  } catch (err) {
    console.error("Comment submission failed", err);
  } finally {
    closeFeedbackModal();
  }
}

/* ==========================================================================
   Corpus Explorer
   ========================================================================== */
async function loadDocuments() {
  const grid = document.getElementById("documents-grid");
  if (!grid) return;

  try {
    const res = await fetch("/api/v1/documents");
    if (!res.ok) throw new Error("Could not retrieve documents");
    const data = await res.json();
    allLoadedDocuments = data.documents || [];

    // Update Metric Cards
    const docsCountEl = document.getElementById("metric-docs-count");
    const chunksCountEl = document.getElementById("metric-chunks-count");
    if (docsCountEl) docsCountEl.innerText = data.total_documents;
    if (chunksCountEl) chunksCountEl.innerText = data.total_vectors;

    renderDocumentCards(allLoadedDocuments);
  } catch (err) {
    grid.innerHTML = `<div style="color: var(--accent-rose); grid-column: 1/-1;">Error loading corpus: ${escapeHtml(err.message)}</div>`;
  }
}

function renderDocumentCards(docs) {
  const grid = document.getElementById("documents-grid");
  if (!grid) return;

  if (docs.length === 0) {
    grid.innerHTML = `<div style="color: var(--text-dim); grid-column: 1/-1; padding: 2rem 0; text-align: center;">No documents match your search criteria.</div>`;
    return;
  }

  grid.innerHTML = docs.map((doc) => {
    const cleanSource = doc.source.split(/[\\/]/).pop() || doc.source;
    return `
      <div class="doc-card">
        <div>
          <div class="doc-header-row">
            <span class="doc-title">📖 ${escapeHtml(doc.title || cleanSource)}</span>
            <span class="doc-type-badge">${escapeHtml(doc.source_type || 'file')}</span>
          </div>
          <div class="doc-path" style="margin-top: 0.35rem;">
            ${escapeHtml(doc.source)}
          </div>
        </div>

        <div class="doc-meta-row">
          <span>📦 <strong>${doc.chunk_count}</strong> indexed chunks</span>
          <button class="doc-query-action-btn" onclick="queryAboutDoc('${escapeJs(doc.title || cleanSource)}')">
            Ask Question ➔
          </button>
        </div>
      </div>
    `;
  }).join("");
}

function filterDocuments() {
  const searchInput = document.getElementById("corpus-search");
  const query = searchInput ? searchInput.value.toLowerCase().trim() : "";
  if (!query) {
    renderDocumentCards(allLoadedDocuments);
    return;
  }

  const filtered = allLoadedDocuments.filter((d) => 
    (d.title && d.title.toLowerCase().includes(query)) ||
    (d.source && d.source.toLowerCase().includes(query))
  );

  renderDocumentCards(filtered);
}

function queryAboutDoc(docName) {
  switchTab("chat");
  const input = document.getElementById("user-input");
  if (input) {
    input.value = `Explain the key concepts and implementation details covered in ${docName}.`;
    input.focus();
  }
}

/* ==========================================================================
   Document Ingestion (File & URL)
   ========================================================================== */
function setPresetUrl(url) {
  const urlInput = document.getElementById("url-input");
  if (urlInput) {
    urlInput.value = url;
    urlInput.focus();
  }
}

async function handleFileUpload(event) {
  const files = event.target.files;
  if (!files || files.length === 0) return;

  const statusBox = document.getElementById("ingest-status-box");
  statusBox.style.display = "flex";
  statusBox.style.background = "rgba(99, 102, 241, 0.12)";
  statusBox.style.border = "1px solid rgba(99, 102, 241, 0.3)";
  statusBox.style.color = "#c7d2fe";
  statusBox.innerHTML = `
    <span class="spin">⚡</span>
    <span>Uploading and executing hierarchical chunker on ${files.length} document(s)...</span>
  `;

  const formData = new FormData();
  for (let i = 0; i < files.length; i++) {
    formData.append("files", files[i]);
  }

  try {
    const res = await fetch("/api/v1/ingest/upload", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "File ingestion failed");

    statusBox.style.background = "rgba(16, 185, 129, 0.12)";
    statusBox.style.border = "1px solid rgba(16, 185, 129, 0.3)";
    statusBox.style.color = "#6ee7b7";
    statusBox.innerHTML = `✅ Successfully indexed: ${escapeHtml(data.message)}`;
    
    showToast(`Indexed ${files.length} document(s) into ChromaDB`, "success");
    checkHealth();
    loadDocuments();
  } catch (err) {
    statusBox.style.background = "rgba(244, 63, 94, 0.12)";
    statusBox.style.border = "1px solid rgba(244, 63, 94, 0.3)";
    statusBox.style.color = "#fda4af";
    statusBox.innerHTML = `❌ Ingestion Error: ${escapeHtml(err.message)}`;
    showToast("Ingestion failed: " + err.message, "error");
  }
}

async function handleUrlIngest() {
  const urlInput = document.getElementById("url-input");
  const btn = document.getElementById("ingest-url-btn");
  const url = urlInput ? urlInput.value.trim() : "";
  if (!url) return;

  const statusBox = document.getElementById("ingest-status-box");
  statusBox.style.display = "flex";
  statusBox.style.background = "rgba(99, 102, 241, 0.12)";
  statusBox.style.border = "1px solid rgba(99, 102, 241, 0.3)";
  statusBox.style.color = "#c7d2fe";
  statusBox.innerHTML = `
    <span class="spin">⚡</span>
    <span>Crawling, sanitizing, and embedding documentation from ${escapeHtml(url)}...</span>
  `;
  btn.disabled = true;

  try {
    const res = await fetch("/api/v1/ingest", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ urls: [url] }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "URL crawler failed");

    statusBox.style.background = "rgba(16, 185, 129, 0.12)";
    statusBox.style.border = "1px solid rgba(16, 185, 129, 0.3)";
    statusBox.style.color = "#6ee7b7";
    statusBox.innerHTML = `✅ ${escapeHtml(data.message)}`;
    urlInput.value = "";
    
    showToast("Successfully crawled and indexed URL", "success");
    checkHealth();
    loadDocuments();
  } catch (err) {
    statusBox.style.background = "rgba(244, 63, 94, 0.12)";
    statusBox.style.border = "1px solid rgba(244, 63, 94, 0.3)";
    statusBox.style.color = "#fda4af";
    statusBox.innerHTML = `❌ Crawl Error: ${escapeHtml(err.message)}`;
    showToast("URL crawl failed: " + err.message, "error");
  } finally {
    btn.disabled = false;
  }
}

/* ==========================================================================
   Markdown Formatter & Code Copy Utilities
   ========================================================================== */
function renderMarkdownWithCode(rawText) {
  if (!rawText) return "";

  // Render fenced code separately. All user/model-provided text is escaped
  // before adding the small, fixed set of presentation tags below.
  const source = String(rawText);
  const codePattern = /```([a-zA-Z0-9_\-\.]*)\n([\s\S]*?)```/g;
  let cursor = 0;
  let output = "";
  let match;

  while ((match = codePattern.exec(source)) !== null) {
    output += renderSafeMarkdownText(source.slice(cursor, match.index));

    const language = match[1].trim() || "code";
    const cleanCode = match[2].trim();
    output += `
      <div class="code-block-wrapper">
        <div class="code-header">
          <span>${escapeHtml(language)}</span>
          <button type="button" class="copy-code-btn" onclick="copySnippet(this)">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg>
            <span>Copy</span>
          </button>
        </div>
        <pre><code class="language-${escapeHtml(language)}">${escapeHtml(cleanCode)}</code></pre>
      </div>
    `;
    cursor = codePattern.lastIndex;
  }

  output += renderSafeMarkdownText(source.slice(cursor));
  return output;
}

function renderSafeMarkdownText(rawText) {
  // Escape raw HTML first. Formatting rules only introduce trusted tags;
  // retrieved documents and model output can never contribute executable HTML.
  return escapeHtml(rawText)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/\*([^*]+)\*/g, "<em>$1</em>")
    .replace(/^#### (.*$)/gim, "<h4>$1</h4>")
    .replace(/^### (.*$)/gim, "<h3>$1</h3>")
    .replace(/^## (.*$)/gim, "<h2>$1</h2>")
    .replace(/^# (.*$)/gim, "<h1>$1</h1>")
    .replace(/^\s*[-*]\s+(.*$)/gim, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gims, "<ul>$1</ul>")
    .replace(/\n\n/g, "<p></p>")
    .replace(/\n/g, "<br/>");
}

function copySnippet(btn) {
  const pre = btn.closest(".code-block-wrapper").querySelector("pre code");
  if (!pre) return;
  const text = pre.innerText;

  navigator.clipboard.writeText(text).then(() => {
    const span = btn.querySelector("span");
    const origText = span ? span.innerText : "Copy";
    if (span) span.innerText = "Copied!";
    btn.style.color = "#34d399";
    btn.style.borderColor = "#34d399";

    setTimeout(() => {
      if (span) span.innerText = origText;
      btn.style.color = "";
      btn.style.borderColor = "";
    }, 2000);
  });
}

function copyFullAnswer(btn) {
  const bubble = btn.closest(".bot-bubble");
  const markdownBody = bubble.querySelector(".markdown-body");
  if (!markdownBody) return;

  navigator.clipboard.writeText(markdownBody.innerText).then(() => {
    showToast("Answer copied to clipboard!", "info");
  });
}

/* ==========================================================================
   Toast Notification System
   ========================================================================== */
function showToast(message, type = "info", duration = 3000) {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;

  const icon = type === "success" ? "✅" : (type === "error" ? "❌" : "ℹ️");
  toast.innerHTML = `
    <span>${icon}</span>
    <span>${escapeHtml(message)}</span>
  `;

  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateY(10px)";
    toast.style.transition = "all 0.25s ease";
    setTimeout(() => toast.remove(), 250);
  }, duration);
}

/* ==========================================================================
   Empty State Template
   ========================================================================== */
/* ==========================================================================
   Application Download Modal & Session Export Controls
   ========================================================================== */
function openDownloadModal() {
  const modal = document.getElementById("download-modal");
  if (modal) modal.style.display = "flex";
}

function closeDownloadModal() {
  const modal = document.getElementById("download-modal");
  if (modal) modal.style.display = "none";
}

function notifyDownload(pkgName) {
  showToast(`Starting download: ${pkgName}... Run the installer to begin!`, "success");
  setTimeout(() => {
    closeDownloadModal();
  }, 1000);
}

function exportCurrentSession() {
  if (!currentSessionId) return;
  showToast("Exporting conversation session transcript...", "info");
  window.location.href = `/api/v1/query/export/${encodeURIComponent(currentSessionId)}?format=markdown`;
}

async function clearCurrentSession() {
  if (!confirm("Are you sure you want to clear this conversation history?")) return;
  try {
    const res = await fetch(`/api/v1/query/history/${encodeURIComponent(currentSessionId)}`, {
      method: "DELETE"
    });
    if (!res.ok) throw new Error("Could not clear session history");
    const container = document.getElementById("chat-messages");
    if (container) container.innerHTML = getEmptyStateHtml();
    showToast("Conversation cleared", "info");
    loadRecentSessions();
  } catch (err) {
    showToast("Failed to clear conversation: " + err.message, "error");
  }
}

function getEmptyStateHtml() {
  return `
    <div class="empty-state" id="empty-state">
      <div class="hero-halo">
        <div class="hero-icon-box">
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
            <polyline points="2 17 12 22 22 17"></polyline>
            <polyline points="2 12 12 17 22 12"></polyline>
          </svg>
        </div>
      </div>

      <h1 class="hero-title">Technical Documentation Assistant</h1>
      <p class="hero-desc">
        High-precision, hallucination-resistant engineering knowledge base. Powered by self-corrective query expansion, document grading, and granular section citations.
      </p>

      <div class="feature-badges">
        <span class="feature-pill">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#34d399" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
          Self-RAG Grounded
        </span>
        <span class="feature-pill">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
          Query Reformulation
        </span>
        <span class="feature-pill">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="#c084fc" stroke-width="2.5"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
          Granular Markdown Citations
        </span>
      </div>

      <!-- Desktop App 1-Click Download Callout -->
      <div class="desktop-banner-card">
        <div class="banner-left">
          <div class="banner-icon">💻</div>
          <div>
            <div class="banner-title">Run DocuGraph AI as a Native Windows App</div>
            <div class="banner-sub">Standalone desktop app with offline WebView2, embedded ChromaDB, and FastEmbed.</div>
          </div>
        </div>
        <div class="banner-actions">
          <a href="/download/exe" class="banner-download-btn" onclick="notifyDownload('Windows Setup (.exe)')">
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
            Download .exe
          </a>
          <button class="banner-outline-btn" onclick="openDownloadModal()">
            All Packages
          </button>
        </div>
      </div>

      <div class="suggestions-container">
        <div class="suggestions-label">Try asking about:</div>
        <div class="suggestions-grid">
          <button class="chip-card" onclick="sendSuggestion('How do I use yield in FastAPI dependencies for resource cleanup and rollback?')">
            <div class="chip-icon">⚡</div>
            <div class="chip-text">
              <span class="chip-headline">FastAPI Yield Dependencies</span>
              <span class="chip-sub">Resource teardown & context managers</span>
            </div>
          </button>

          <button class="chip-card" onclick="sendSuggestion('What is LangGraph StateGraph and how do conditional edges control workflow routing?')">
            <div class="chip-icon">🔄</div>
            <div class="chip-text">
              <span class="chip-headline">LangGraph Conditional Edges</span>
              <span class="chip-sub">StateGraph routing & bounded retry cycles</span>
            </div>
          </button>

          <button class="chip-card" onclick="sendSuggestion('How does Pydantic v2 use @field_validator to validate fields with mode=\\'before\\' or \\'after\\'?')">
            <div class="chip-icon">🛡️</div>
            <div class="chip-text">
              <span class="chip-headline">Pydantic v2 @field_validator</span>
              <span class="chip-sub">Schema validation & data transformation</span>
            </div>
          </button>

          <button class="chip-card" onclick="sendSuggestion('Explain REST API idempotency, safe methods, and appropriate HTTP status codes.')">
            <div class="chip-icon">🌐</div>
            <div class="chip-text">
              <span class="chip-headline">REST API Idempotency Rules</span>
              <span class="chip-sub">PUT vs POST semantics & status codes</span>
            </div>
          </button>
        </div>
      </div>
    </div>
  `;
}

/* ==========================================================================
   Escaping Utilities
   ========================================================================== */
function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeJs(text) {
  if (!text) return "";
  return String(text).replace(/'/g, "\\'").replace(/"/g, '\\"');
}
