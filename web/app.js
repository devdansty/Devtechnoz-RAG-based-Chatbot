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
const conversationStorageKey = "devtechnoz-chat-conversation-id";
const uuidPattern = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-8][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const acknowledgmentPattern =
  /^(?:yes|yeah|yep|sure|ok(?:ay)?|alright|no|nope)(?:\s*,?\s*please)?[.!?]*$|^(?:thanks|thank you|go ahead|please do|that works|sounds good)[.!?]*$/i;
const greetingPattern =
  /^(?:hi|hello|hey|good morning|good afternoon|good evening)[.!?]*$/i;
let conversationId = localStorage.getItem(conversationStorageKey);
if (!conversationId || !uuidPattern.test(conversationId)) {
  conversationId = crypto.randomUUID();
  localStorage.setItem(conversationStorageKey, conversationId);
}
let isSending = false;
let latestQuestion = "";

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

function renderContactLinks(bubble, text, question) {
  const phonePattern = /(?:\+?92[\s().-]*309[\s().-]*1404[\s().-]*002)/gi;
  const hasWhatsAppNumber = phonePattern.test(text);
  phonePattern.lastIndex = 0;
  const displayText = hasWhatsAppNumber
    ? text
        .replace(phonePattern, "")
        .replace(/\bat\s*(?=[,.;!?]|$)/gi, "")
        .replace(/\s+([,.;!?])/g, "$1")
    : text;
  const emailPattern = /\b(?:info|contact)@devtechnoz\.com\b/gi;
  let cursor = 0;
  bubble.replaceChildren();

  for (const match of displayText.matchAll(emailPattern)) {
    const contact = match[0];
    const start = match.index;
    bubble.append(document.createTextNode(displayText.slice(cursor, start)));

    const link = document.createElement("a");
    link.textContent = contact;
    link.href = `mailto:${contact}`;
    bubble.append(link);
    cursor = start + contact.length;
  }

  bubble.append(document.createTextNode(displayText.slice(cursor)));
  appendWhatsAppHandoff(bubble, text, question, hasWhatsAppNumber);
}

function appendWhatsAppHandoff(bubble, text, question, hasWhatsAppNumber) {
  const mentionsPerson = /\b(?:real person|human(?: agent| support)?|live agent|representative|support team|someone from our team|our team)\b/i.test(text);
  const mentionsContact = /\b(?:WhatsApp|talk|speak|chat|connect|continue|message|contact|reach out)\b/i.test(text);
  const mentionsWhatsAppOffer = /\bWhatsApp\b/i.test(text) && /\b(?:connect|continue|message|contact|reach out|schedule|call)\b/i.test(text);
  if (
    (!hasWhatsAppNumber && !mentionsWhatsAppOffer && !(mentionsPerson && mentionsContact)) ||
    !hasExplicitHandoffIntent(question)
  ) {
    return;
  }

  const button = document.createElement("button");
  button.type = "button";
  button.className = "whatsapp-handoff-button";
  button.disabled = true;
  button.textContent = "Preparing a WhatsApp message...";
  button.setAttribute("aria-label", "Continue this conversation on WhatsApp with a real person");
  bubble.append(button);
  button.addEventListener("click", () => void prepareWhatsAppMessage(button));
  void prepareWhatsAppMessage(button);
}

function hasExplicitHandoffIntent(question) {
  const normalizedQuestion = question.trim();
  if (
    /^(?:whatsapp|a real person|real person|a person|person|someone|human|human agent|representative|support team|our team)[.!?]*$/i.test(
      normalizedQuestion,
    )
  ) {
    return true;
  }

  const directRequest =
    /\b(?:i want|i need|i(?:'d| would) like|please|can i|could i|can you|could you|connect me|put me in touch|talk to|speak to|chat with|message your|contact your|call)\b[\s\S]{0,100}\b(?:whatsapp|person|human(?: agent| support)?|live agent|representative|support team|your team|someone)\b/i;
  const directWhatsAppRequest =
    /\bwhatsapp\b/i.test(normalizedQuestion) &&
    /\b(?:want|need|please|send|message|contact|connect|use|number)\b/i.test(
      normalizedQuestion,
    );
  if (directRequest.test(normalizedQuestion) || directWhatsAppRequest) return true;
  if (!acknowledgmentPattern.test(normalizedQuestion)) return false;

  const previousAssistant = conversation
    .slice(0, -2)
    .reverse()
    .find(([role]) => role === "assistant")?.[1];
  if (!previousAssistant) return false;

  const previousQuestions = [...previousAssistant.matchAll(/[^?]*\?/g)];
  const lastQuestion = previousQuestions.length
    ? previousQuestions[previousQuestions.length - 1][0]
    : "";
  return (
    /\b(?:would you like|do you want|should i|can i|could i|may i)\b/i.test(lastQuestion) &&
    /\b(?:real person|person|human|agent|representative|support team|our team|someone|whatsapp)\b/i.test(lastQuestion) &&
    /\b(?:connect|talk|speak|chat|contact|message)\b/i.test(lastQuestion) &&
    !/\bor\b/i.test(lastQuestion)
  );
}

async function prepareWhatsAppMessage(button) {
  const previousError = button.nextElementSibling;
  if (previousError?.classList.contains("whatsapp-handoff-error")) {
    previousError.remove();
  }
  button.disabled = true;
  button.textContent = "Preparing a WhatsApp message...";

  try {
    const response = await fetch("/api/whatsapp-message", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ history: recentHistory() }),
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(
        payload.error || `WhatsApp message request failed (${response.status}).`,
      );
    }
    if (typeof payload.message !== "string" || !payload.message.trim()) {
      throw new Error("The server returned an empty WhatsApp message.");
    }

    const link = document.createElement("a");
    link.className = "whatsapp-handoff-button";
    link.href = `https://wa.me/923091404002?text=${encodeURIComponent(payload.message.trim())}`;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.textContent = "Continue on WhatsApp with a real person";
    link.setAttribute(
      "aria-label",
      "Continue this conversation on WhatsApp with a real person",
    );
    button.replaceWith(link);
  } catch (error) {
    button.disabled = false;
    button.textContent = "Retry WhatsApp message";
    const errorMessage = document.createElement("span");
    errorMessage.className = "whatsapp-handoff-error";
    errorMessage.setAttribute("role", "alert");
    errorMessage.textContent =
      error instanceof Error
        ? `Couldn't prepare the message. ${error.message}`
        : "Couldn't prepare the message. Please try again.";
    button.after(errorMessage);
  }
}

async function restoreSavedConversation() {
  const response = await fetch("/api/chat/history", {
    headers: { "X-Conversation-ID": conversationId },
  });
  if (!response.ok) {
    const problem = await response.json().catch(() => ({}));
    throw new Error(problem.error || `History request failed (${response.status}).`);
  }

  const payload = await response.json();
  if (!payload.enabled || !Array.isArray(payload.messages)) return;
  for (const message of payload.messages) {
    if (
      !message ||
      !["human", "assistant"].includes(message.role) ||
      typeof message.content !== "string"
    ) {
      throw new Error("The saved conversation contains an invalid message.");
    }
    conversation.push([message.role, message.content]);
    const bubble = addMessage(message.role, message.content);
    if (message.role === "human") latestQuestion = message.content;
    else renderContactLinks(bubble, message.content, latestQuestion);
  }
  if (conversation.length > 20) conversation.splice(0, conversation.length - 20);
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
    body: JSON.stringify({
      conversation_id: conversationId,
      question,
      history: recentHistory(),
    }),
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

  setBusy(true);
  let replyBubble;

  try {
    await historyReady;
    latestQuestion = question;
    quickPrompts.hidden = true;
    addMessage("human", question);
    replyBubble = addMessage("assistant", "");
    replyBubble.classList.add("streaming");

    const answer = await streamReply(question, replyBubble);
    conversation.push(["human", question], ["assistant", answer]);
    if (conversation.length > 20) conversation.splice(0, conversation.length - 20);
    if (answer) renderContactLinks(replyBubble, answer, question);
    else replyBubble.textContent = "I couldn't find an answer just now. Please try again.";
  } catch (error) {
    if (replyBubble) {
      replyBubble.classList.remove("streaming");
      replyBubble.textContent = replyBubble.textContent
        ? `${replyBubble.textContent}\n\n${error.message}`
        : `Sorry, I couldn't complete that reply. ${error.message}`;
      replyBubble.setAttribute("role", "alert");
    } else {
      const errorBubble = addMessage(
        "assistant",
        `Sorry, I couldn't complete that reply. ${error.message}`,
      );
      errorBubble.setAttribute("role", "alert");
    }
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

const historyReady = restoreSavedConversation().catch((error) => {
  const warning = addMessage(
    "assistant",
    `I couldn't restore your previous conversation. You can continue chatting here. (${error.message})`,
  );
  warning.setAttribute("role", "alert");
});
