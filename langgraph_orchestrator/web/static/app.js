/* ============================================================
   Dermatology Chatbot — Frontend Logic
   ============================================================ */

"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let sessionId = null;
let pendingImageFile = null;  // File object waiting to be uploaded

// ---------------------------------------------------------------------------
// DOM refs (resolved once on load)
// ---------------------------------------------------------------------------

const chatMessages    = document.getElementById("chat-messages");
const messageInput    = document.getElementById("message-input");
const sendBtn         = document.getElementById("send-btn");
const uploadBtn       = document.getElementById("upload-btn");
const imageFileInput  = document.getElementById("image-file-input");
const uploadThumb     = document.getElementById("upload-thumb");
const uploadFilename  = document.getElementById("upload-filename");
const resetBtn        = document.getElementById("reset-btn");
const phaseBadge      = document.getElementById("phase-badge");
const nodePath        = document.getElementById("node-path");
const slotGrid        = document.getElementById("slot-grid");
const flagsRow        = document.getElementById("flags-row");
const stateJson       = document.getElementById("state-json");
const densenetSection = document.getElementById("densenet-section");
const densenetBody    = document.getElementById("densenet-body");
const diagSection     = document.getElementById("diag-section");
const diagBody        = document.getElementById("diag-body");
const toastEl         = document.getElementById("toast");

// ---------------------------------------------------------------------------
// Initialisation
// ---------------------------------------------------------------------------

async function init() {
  try {
    const res  = await fetch("/session/start", { method: "POST" });
    const data = await res.json();
    sessionId  = data.session_id;
    updatePhaseBadge(data.phase);
    appendSystemMessage("Session started. Describe your skin concern to begin.");
    console.log("[init] session_id =", sessionId);
  } catch (err) {
    appendSystemMessage("Could not connect to server. Is the backend running?");
    console.error("[init]", err);
  }
}

// ---------------------------------------------------------------------------
// Send message
// ---------------------------------------------------------------------------

async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || !sessionId) return;

  messageInput.value = "";
  adjustTextareaHeight();

  // If a pending image hasn't been uploaded yet, upload it first
  if (pendingImageFile) {
    await uploadPendingImage();
  }

  addMessageBubble("user", text);
  showTyping();
  disableInput(true);

  try {
    const res  = await fetch("/chat/message", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ session_id: sessionId, message: text }),
    });
    const data = await res.json();

    hideTyping();
    addMessageBubble("assistant", data.response);
    updateDebugFromTurn(data);
    await refreshDebugPanel();
  } catch (err) {
    hideTyping();
    addMessageBubble("assistant", "[Error] Could not reach the server.");
    console.error("[sendMessage]", err);
  } finally {
    disableInput(false);
    messageInput.focus();
  }
}

// ---------------------------------------------------------------------------
// Image upload
// ---------------------------------------------------------------------------

function onFileSelected(event) {
  const file = event.target.files[0];
  if (!file) return;

  pendingImageFile = file;

  // Show thumbnail + filename above the input
  const reader = new FileReader();
  reader.onload = (e) => {
    uploadThumb.src     = e.target.result;
    uploadThumb.style.display = "block";
  };
  reader.readAsDataURL(file);
  uploadFilename.textContent = file.name;
  showToast("Image ready — send a message to upload it.");
}

async function uploadPendingImage() {
  if (!pendingImageFile || !sessionId) return;

  const formData = new FormData();
  formData.append("session_id", sessionId);
  formData.append("file",       pendingImageFile);

  // Show preview in chat
  const previewUrl = URL.createObjectURL(pendingImageFile);
  addImagePreviewToChat(previewUrl);

  // Clear pending state
  pendingImageFile = null;
  uploadThumb.style.display = "none";
  uploadFilename.textContent = "";
  imageFileInput.value = "";

  try {
    const res  = await fetch("/chat/upload-image", { method: "POST", body: formData });
    const data = await res.json();

    addMessageBubble("assistant", data.response);
    updateDebugFromTurn(data);
    await refreshDebugPanel();
  } catch (err) {
    console.error("[uploadPendingImage]", err);
    addMessageBubble("assistant", "[Error] Image upload failed.");
  }
}

// ---------------------------------------------------------------------------
// Reset session
// ---------------------------------------------------------------------------

async function resetSession() {
  if (!sessionId) return;
  if (!confirm("Reset the session? This clears all conversation history.")) return;

  try {
    await fetch("/session/reset", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ session_id: sessionId }),
    });

    // Clear UI
    chatMessages.innerHTML = "";
    pendingImageFile = null;
    uploadThumb.style.display = "none";
    uploadFilename.textContent = "";
    imageFileInput.value = "";

    appendSystemMessage("Session reset. Start a new conversation.");
    updatePhaseBadge("start");
    clearDebugPanel();
    showToast("Session reset.");
  } catch (err) {
    console.error("[resetSession]", err);
    showToast("Reset failed — check the server.");
  }
}

// ---------------------------------------------------------------------------
// Debug panel — incremental update (from turn response)
// ---------------------------------------------------------------------------

function updateDebugFromTurn(data) {
  updatePhaseBadge(data.phase);

  // Node path
  nodePath.innerHTML = "";
  (data.nodes_visited || []).forEach((name, i) => {
    if (i > 0) {
      const arrow = document.createElement("span");
      arrow.className = "node-arrow";
      arrow.textContent = "→";
      nodePath.appendChild(arrow);
    }
    const chip = document.createElement("span");
    chip.className   = "node-chip";
    chip.textContent = name;
    nodePath.appendChild(chip);
  });

  // Flags
  renderFlags({
    symptoms_complete: data.symptoms_complete,
    image_uploaded:    data.image_uploaded,
    diagnosis_done:    data.diagnosis_done,
    rag_context_used:  data.rag_context_used,
  });
}

// ---------------------------------------------------------------------------
// Debug panel — full refresh (from GET /session/{id}/state)
// ---------------------------------------------------------------------------

async function refreshDebugPanel() {
  if (!sessionId) return;

  try {
    const res   = await fetch(`/session/${sessionId}/state`);
    const state = await res.json();

    // Symptom slots
    renderSlots(state.symptom_slots || {});

    // Flags (full set)
    renderFlags({
      symptoms_complete:  state.symptoms_complete,
      image_uploaded:     state.image_uploaded,
      image_processed:    state.image_processed,
      diagnosis_done:     state.diagnosis_done,
      fallback_triggered: state.fallback_triggered,
      uncertainty_flag:   state.uncertainty_flag,
    });

    // DenseNet output
    if (state.densenet_output) {
      densenetSection.style.display = "block";
      const d = state.densenet_output;
      densenetBody.innerHTML = `
        <div class="densenet-row">
          <span class="densenet-key">Label</span>
          <span>${esc(d.label || "—")}</span>
          <span class="densenet-key">Confidence</span>
          <span>${d.confidence !== undefined ? (d.confidence * 100).toFixed(1) + "%" : "—"}</span>
          <span class="densenet-key">Morphology</span>
          <span>${esc(d.morphology || "—")}</span>
        </div>`;
    } else {
      densenetSection.style.display = "none";
    }

    // Diagnosis summary
    if (state.diagnosis_summary) {
      diagSection.style.display = "block";
      diagBody.textContent = state.diagnosis_summary;
    } else {
      diagSection.style.display = "none";
    }

    // Raw JSON — exclude messages for readability
    const { messages, ...rest } = state;
    rest.message_count = messages ? messages.length : 0;
    stateJson.textContent = JSON.stringify(rest, null, 2);

  } catch (err) {
    console.error("[refreshDebugPanel]", err);
  }
}

// ---------------------------------------------------------------------------
// Debug panel — helpers
// ---------------------------------------------------------------------------

function renderSlots(slots) {
  const keys = ["location", "duration", "appearance", "severity", "triggers"];
  slotGrid.innerHTML = "";

  keys.forEach(k => {
    const kEl = document.createElement("span");
    kEl.className   = "slot-key";
    kEl.textContent = k;

    const vEl = document.createElement("span");
    vEl.className   = slots[k] ? "slot-value" : "slot-value empty";
    vEl.textContent = slots[k] || "—";

    slotGrid.appendChild(kEl);
    slotGrid.appendChild(vEl);
  });
}

function renderFlags(flags) {
  flagsRow.innerHTML = "";
  Object.entries(flags).forEach(([key, val]) => {
    const chip = document.createElement("span");
    chip.className   = `flag-chip ${val ? "true" : "false"}`;
    chip.textContent = `${key.replace(/_/g, " ")}: ${val ? "✓" : "✗"}`;
    flagsRow.appendChild(chip);
  });
}

function clearDebugPanel() {
  nodePath.innerHTML = "";
  slotGrid.innerHTML = "";
  flagsRow.innerHTML = "";
  stateJson.textContent = "{}";
  densenetSection.style.display = "none";
  diagSection.style.display     = "none";
}

// ---------------------------------------------------------------------------
// Chat UI helpers
// ---------------------------------------------------------------------------

function addMessageBubble(role, text) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;

  const label = document.createElement("div");
  label.className   = "message-label";
  label.textContent = role === "user" ? "You" : "Assistant";

  const bubble = document.createElement("div");
  bubble.className   = "bubble";
  bubble.textContent = text;

  wrapper.appendChild(label);
  wrapper.appendChild(bubble);
  chatMessages.appendChild(wrapper);
  scrollToBottom();
}

function addImagePreviewToChat(src) {
  const wrapper = document.createElement("div");
  wrapper.className = "message user";

  const label = document.createElement("div");
  label.className   = "message-label";
  label.textContent = "You — image";

  const img = document.createElement("img");
  img.className = "chat-image-preview";
  img.src       = src;
  img.alt       = "Uploaded skin image";

  wrapper.appendChild(label);
  wrapper.appendChild(img);
  chatMessages.appendChild(wrapper);
  scrollToBottom();
}

function appendSystemMessage(text) {
  addMessageBubble("assistant", text);
}

function showTyping() {
  const el = document.createElement("div");
  el.id        = "typing-indicator";
  el.className = "typing-indicator";
  el.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
  chatMessages.appendChild(el);
  scrollToBottom();
}

function hideTyping() {
  const el = document.getElementById("typing-indicator");
  if (el) el.remove();
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ---------------------------------------------------------------------------
// Phase badge
// ---------------------------------------------------------------------------

function updatePhaseBadge(phase) {
  phaseBadge.textContent          = (phase || "—").replace(/_/g, " ");
  phaseBadge.dataset.phase        = phase || "";
}

// ---------------------------------------------------------------------------
// Input state
// ---------------------------------------------------------------------------

function disableInput(disabled) {
  messageInput.disabled = disabled;
  sendBtn.disabled      = disabled;
  uploadBtn.disabled    = disabled;
}

function adjustTextareaHeight() {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 120) + "px";
}

// ---------------------------------------------------------------------------
// Toast notification
// ---------------------------------------------------------------------------

let toastTimer = null;

function showToast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), 2800);
}

// ---------------------------------------------------------------------------
// XSS helper
// ---------------------------------------------------------------------------

function esc(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

sendBtn.addEventListener("click", sendMessage);

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

messageInput.addEventListener("input", adjustTextareaHeight);

uploadBtn.addEventListener("click", () => imageFileInput.click());

imageFileInput.addEventListener("change", onFileSelected);

resetBtn.addEventListener("click", resetSession);

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", init);
