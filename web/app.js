const launcher = document.querySelector("#chat-launcher");
const panel = document.querySelector("#chat-panel");
const closeButton = document.querySelector("#chat-close");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-button");
const messages = document.querySelector("#chat-messages");
const quickPrompts = document.querySelector("#quick-prompts");
const status = document.createElement("div");
const conversation = [];
let isSending = false;

status.className = "chat-status";
status.hidden = true;
status.innerHTML = '<span class="typing-dots" aria-hidden="true"><i></i><i></i><i></i></span><span>Finding the right details...</span>';
messages.after(status);

function setChatOpen(isOpen) {
  panel.hidden = !isOpen;
  launcher.setAttribute("aria-expanded", String(isOpen));
  launcher.setAttribute("aria-label", isOpen ? "Close DevTechNoz chat" : "Open DevTechNoz chat");
  if (isOpen) input.focus();
  else launcher.focus();
}

function recentHistory() {
  let remainingCharacters = 16_000;
  const recent = [];
  for (const [role, content] of conversation.slice(-20).reverse()) {
    if (remainingCharacters <= 0) break;
    const shortened = content.slice(-remainingCharacters);
    recent.unshift([role, shortened]);
    remainingCharacters -= shortened.length;
  }
  return recent;
}

function scrollToLatest() {
  messages.scrollTop = messages.scrollHeight;
}

function addMessage(role, text) {
  const row = document.createElement("div");
  row.className = `message-row ${role === "human" ? "user-message" : "assistant-message"}`;
  if (role !== "human") {
    const avatar = document.createElement("span");
    avatar.className = "message-avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = "D";
    row.append(avatar);
  }

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.textContent = text;
  row.append(bubble);
  messages.append(row);
  scrollToLatest();
  return bubble;
}

function setBusy(isBusy) {
  isSending = isBusy;
  sendButton.disabled = isBusy;
  input.disabled = isBusy;
  form.setAttribute("aria-busy", String(isBusy));
  status.hidden = !isBusy;
}

function parseEvent(block, bubble, state) {
  let eventName = "message";
  const data = [];
  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      eventName = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data.push(line.slice(5).trimStart());
    }
  }
  if (!data.length) return;

  const payload = JSON.parse(data.join("\n"));
  if (eventName === "error") {
    throw new Error(payload.message || "The assistant could not complete this reply.");
  }
  if (eventName === "done") {
    state.done = true;
    return;
  }
  if (typeof payload.text === "string" && payload.text) {
    state.text += payload.text;
    bubble.textContent = state.text;
    bubble.classList.add("streaming");
    status.hidden = true;
    scrollToLatest();
  }
}

async function streamReply(question, bubble) {
  const response = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, history: recentHistory() }),
  });
  if (!response.ok) {
    const problem = await response.json().catch(() => ({}));
    throw new Error(problem.error || `Chat request failed (${response.status}).`);
  }
  if (!response.body) {
    throw new Error("Streaming is not supported by this browser.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const state = { text: "", done: false };
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      parseEvent(block, bubble, state);
      boundary = buffer.indexOf("\n\n");
    }
    if (done) break;
  }

  if (buffer.trim()) parseEvent(buffer, bubble, state);
  if (!state.done) throw new Error("The connection ended before the reply was complete.");
  bubble.classList.remove("streaming");
  return state.text;
}

async function sendMessage(rawText) {
  const question = rawText.trim();
  if (!question || isSending) return;

  quickPrompts.hidden = true;
  addMessage("human", question);
  const replyBubble = addMessage("assistant", "");
  replyBubble.classList.add("streaming");
  setBusy(true);

  try {
    const answer = await streamReply(question, replyBubble);
    conversation.push(["human", question], ["assistant", answer]);
    if (conversation.length > 20) conversation.splice(0, conversation.length - 20);
    if (!answer) replyBubble.textContent = "I couldn't find an answer just now. Please try again.";
  } catch (error) {
    replyBubble.classList.remove("streaming");
    replyBubble.textContent = replyBubble.textContent
      ? `${replyBubble.textContent}\n\n${error.message}`
      : `Sorry, I couldn't complete that reply. ${error.message}`;
    replyBubble.setAttribute("role", "alert");
  } finally {
    setBusy(false);
    input.value = "";
    input.focus();
    scrollToLatest();
  }
}

launcher.addEventListener("click", () => setChatOpen(panel.hidden));
closeButton.addEventListener("click", () => setChatOpen(false));
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !panel.hidden) setChatOpen(false);
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void sendMessage(input.value);
});

quickPrompts.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-prompt]");
  if (button) void sendMessage(button.dataset.prompt);
});

const menuToggle = document.querySelector("#menu-toggle");
const primaryNav = document.querySelector("#primary-nav");
menuToggle.addEventListener("click", () => {
  const isOpen = primaryNav.classList.toggle("is-open");
  menuToggle.setAttribute("aria-expanded", String(isOpen));
  menuToggle.setAttribute("aria-label", isOpen ? "Close navigation" : "Open navigation");
});

primaryNav.addEventListener("click", (event) => {
  if (event.target.closest("a")) {
    primaryNav.classList.remove("is-open");
    menuToggle.setAttribute("aria-expanded", "false");
    menuToggle.setAttribute("aria-label", "Open navigation");
  }
});
