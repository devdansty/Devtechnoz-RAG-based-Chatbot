const loginCard = document.querySelector("#login-card");
const loginForm = document.querySelector("#login-form");
const passwordInput = document.querySelector("#admin-password");
const loginStatus = document.querySelector("#login-status");
const dashboard = document.querySelector("#dashboard");
const dashboardStatus = document.querySelector("#dashboard-status");
const conversationItems = document.querySelector("#conversation-items");
const loadMoreButton = document.querySelector("#load-more-button");
const transcriptTitle = document.querySelector("#transcript-title");
const transcriptMessages = document.querySelector("#transcript-messages");
const deleteAllButton = document.querySelector("#delete-all-button");
const logoutButton = document.querySelector("#logout-button");
let authorization = "";
let selectedConversationId = "";
let conversationOffset = 0;
const conversationPageSize = 100;

function setStatus(element, message) {
  element.textContent = message;
}

function toBase64(value) {
  const bytes = new TextEncoder().encode(value);
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}

async function adminFetch(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: {
      ...options.headers,
      Authorization: authorization,
    },
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.error || `Request failed (${response.status}).`);
  }
  return payload;
}

function showEmptyMessage(container, text) {
  const empty = document.createElement("p");
  empty.className = "empty-state";
  empty.textContent = text;
  container.append(empty);
}

async function loadTranscript(conversationId) {
  const payload = await adminFetch("/api/admin/conversation", {
    headers: { "X-Conversation-ID": conversationId },
  });
  selectedConversationId = conversationId;
  transcriptTitle.textContent = "Conversation transcript";
  transcriptMessages.replaceChildren();
  if (!payload.messages.length) {
    showEmptyMessage(transcriptMessages, "This conversation has no saved messages.");
    return;
  }
  for (const message of payload.messages) {
    const item = document.createElement("article");
    item.className = `transcript-message ${message.role}`;
    const role = document.createElement("strong");
    role.textContent = message.role === "human" ? "Visitor" : "Assistant";
    const content = document.createElement("p");
    content.textContent = message.content;
    item.append(role, content);
    transcriptMessages.append(item);
  }
}

async function loadConversations(loadOlder = false) {
  setStatus(dashboardStatus, "Loading saved conversations...");
  if (!loadOlder) {
    conversationOffset = 0;
    conversationItems.replaceChildren();
  }
  const payload = await adminFetch(
    `/api/admin/conversations?offset=${conversationOffset}`,
  );
  if (!loadOlder && !payload.conversations.length) {
    showEmptyMessage(conversationItems, "There are no saved conversations.");
    transcriptMessages.replaceChildren();
    transcriptTitle.textContent = "No conversations";
    selectedConversationId = "";
  } else {
    for (const conversation of payload.conversations) {
      const row = document.createElement("div");
      row.className = "conversation-item";
      const select = document.createElement("button");
      select.className = "conversation-select";
      select.type = "button";
      const question = document.createElement("strong");
      question.textContent = conversation.latest_question || "New conversation";
      const details = document.createElement("span");
      details.textContent =
        `${new Date(conversation.updated_at).toLocaleString()} · ` +
        `${conversation.message_count} messages`;
      select.append(question, details);
      select.addEventListener("click", async () => {
        try {
          setStatus(dashboardStatus, "");
          await loadTranscript(conversation.id);
        } catch (error) {
          setStatus(dashboardStatus, error.message);
        }
      });

      const deleteButton = document.createElement("button");
      deleteButton.className = "delete-conversation";
      deleteButton.type = "button";
      deleteButton.textContent = "Delete";
      deleteButton.addEventListener("click", async () => {
        if (!window.confirm("Delete this conversation and all its messages?")) return;
        try {
          await adminFetch(
            "/api/admin/conversation",
            {
              method: "DELETE",
              headers: { "X-Conversation-ID": conversation.id },
            },
          );
          if (selectedConversationId === conversation.id) {
            transcriptTitle.textContent = "Select a conversation";
            transcriptMessages.replaceChildren();
            selectedConversationId = "";
          }
          await loadConversations();
        } catch (error) {
          setStatus(dashboardStatus, error.message);
        }
      });
      row.append(select, deleteButton);
      conversationItems.append(row);
    }
    conversationOffset += payload.conversations.length;
    if (!loadOlder && selectedConversationId) {
      const stillExists = payload.conversations.some(
        (conversation) => conversation.id === selectedConversationId,
      );
      if (!stillExists) {
        transcriptTitle.textContent = "Select a conversation";
        transcriptMessages.replaceChildren();
        selectedConversationId = "";
      }
    }
  }
  loadMoreButton.hidden = payload.conversations.length < conversationPageSize;
  setStatus(dashboardStatus, "");
}

loginForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  authorization = `Basic ${toBase64(`admin:${passwordInput.value}`)}`;
  setStatus(loginStatus, "Checking access...");
  try {
    await loadConversations();
    loginCard.hidden = true;
    dashboard.hidden = false;
    passwordInput.value = "";
  } catch (error) {
    authorization = "";
    setStatus(loginStatus, error.message);
  }
});

deleteAllButton.addEventListener("click", async () => {
  if (!window.confirm("Permanently delete every saved conversation?")) return;
  try {
    await adminFetch("/api/admin/conversations", { method: "DELETE" });
    selectedConversationId = "";
    transcriptTitle.textContent = "No conversations";
    transcriptMessages.replaceChildren();
    await loadConversations();
  } catch (error) {
    setStatus(dashboardStatus, error.message);
  }
});

loadMoreButton.addEventListener("click", async () => {
  try {
    await loadConversations(true);
  } catch (error) {
    setStatus(dashboardStatus, error.message);
  }
});

logoutButton.addEventListener("click", () => {
  authorization = "";
  selectedConversationId = "";
  dashboard.hidden = true;
  loginCard.hidden = false;
  setStatus(loginStatus, "");
});
