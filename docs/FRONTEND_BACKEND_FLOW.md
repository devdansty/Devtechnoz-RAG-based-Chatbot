# How the Frontend and Backend Work Together

## One command starts the web app

From the project root, run:

```powershell
python -m devtechnoz_rag.web_frontend
```

Python imports the `devtechnoz_rag.web_frontend` module and runs its `main()`
function. That function loads environment variables from the project-root
`.env` file and starts one `ThreadingHTTPServer`. The server binds to
`127.0.0.1:8000` by default.

The browser is the frontend; the Python process is the backend. The Python
server also serves the frontend files, so there is no separate frontend
development server or Node.js process to start.

## Request flow

```text
Browser
  | GET /
  v
Python web server ----------------------> web/index.html
  | GET /styles.css and /app.js
  +-------------------------------------> web/styles.css and web/app.js

Browser JavaScript
  | GET /api/chat/history (conversation ID in request header)
  v
Python web server
  | restore last 20 messages from PostgreSQL, if configured
  v
Browser JavaScript
  | POST /api/chat
  | JSON: question + recent history + conversation ID
  v
Python web server
  | validates the request
  v
stream_answer()
  |-- direct FAQ match? return the prepared answer
  |
  +-- otherwise retrieve_chunks()
  |     | Pinecone search over-fetches candidates
  |     + MMR selects relevant, less-repetitive chunks
  |
  +-- ChatGroq streams a response using the retrieved context
  |
  v
Server sends text/event-stream events as chunks arrive
  | event: message
  | event: done (or event: error)
  v
Browser reads each event and updates the assistant message
  |
  +--> PostgreSQL stores completed question-and-answer exchange

After explicit human-handoff consent:
Browser JavaScript
  | POST /api/whatsapp-message (up to 20 recent messages)
  v
Python validates the history and handoff intent
  | ChatGroq drafts a short message from the visitor's stated interests
  v
Browser opens a wa.me link with the generated message prefilled
```

## What happens when you send a message

1. The page in [`index.html`](../web/index.html) loads the chat interface.
   It also loads [`app.js`](../web/app.js), which handles the launcher, input,
   conversation context, and streaming display.
2. The browser keeps a random conversation ID in local storage. On page load it
   requests the most recent 20 saved messages for that ID. On submit, it sends
   the new question, that ID, and up to 20 recent messages to `POST /api/chat`.
3. [`web_frontend.py`](../src/devtechnoz_rag/web_frontend.py) validates the
   question and history. It returns an HTTP `text/event-stream` response and
   calls `stream_answer()`.
4. [`rag_chat.py`](../src/devtechnoz_rag/rag_chat.py) checks for a direct FAQ
   answer. Otherwise, it sends the question (and recent context) to
   [`retrieval_pipline.py`](../src/devtechnoz_rag/retrieval_pipline.py).
5. Retrieval searches Pinecone, gets extra candidates, and applies MMR to select
   a relevant and diverse set of chunks. Those chunks are included in the
   prompt sent to Groq.
6. As Groq produces text chunks, Python sends each one as an SSE `message`
   event and flushes it to the browser. The browser reads the response stream
   and appends the received text to the assistant message. These are provider
   text chunks, not guaranteed one-character events.
7. When generation finishes, Python sends a `done` event. If a handled error
   occurs, it sends an `error` event for the browser to display.
8. When PostgreSQL is configured, Python saves each completed question-and-answer
   exchange. The admin interface at `/admin` uses a password-protected API to
   list, read, and delete transcripts.
9. A WhatsApp button appears only after an explicit human-handoff request or a
   clear yes to a direct human-handoff question. The browser then sends its
   latest 20 messages to `POST /api/whatsapp-message`; the server validates the
   consent context and uses the configured Groq model to draft a message from
   the visitor's stated interests. The resulting `wa.me` link opens with that
   draft prefilled. An ambiguous "yes" does not enable the button.

The browser sends recent messages as model context, while PostgreSQL stores the
complete website transcript. These are separate from terminal chat history,
which continues to use local JSON files. Website records have no automatic
expiry and remain until an administrator deletes them. If `DATABASE_URL` is
unset, the website chat remains usable but history is not persisted.

## Useful routes

| Route | Purpose |
|---|---|
| `GET /` | Serves the landing page. |
| `GET /styles.css` and `GET /app.js` | Serves the frontend assets. |
| `GET /api/health` | Returns `{"status": "ok"}` for a deployment health check. |
| `GET /api/chat/history` | Restores a visitor's recent messages using the `X-Conversation-ID` request header. |
| `POST /api/chat` | Accepts a question and streams the answer as SSE. |
| `POST /api/whatsapp-message` | Validates explicit handoff intent and drafts a WhatsApp message from recent chat history with Groq. |
| `GET /admin` | Serves the transcript administration page. |
| `/api/admin/conversations` | Password-protected transcript list and delete-all API. |
| `/api/admin/conversation` | Password-protected transcript read and delete-one API; conversation ID is sent in a request header. |

## Where configuration lives

Locally, the Python backend reads API keys from the project-root `.env` file.
On a hosting provider, set the same values as service environment variables.
The browser never receives the Pinecone or Groq API keys.
The WhatsApp-draft endpoint uses the same server-side Groq credentials and
model as chat. It is called only after the visitor explicitly chooses a human
handoff and receives the latest 20 messages as context.
Set `DATABASE_URL` to enable PostgreSQL transcript storage and
`CHAT_ADMIN_PASSWORD` to enable admin access. The admin username is `admin`;
the password is supplied through the admin page and is not stored in the
browser's local storage.

The web server and the API use the same host and port. Locally, open
`http://127.0.0.1:8000`; on a hosted service, the platform's URL points to this
same Python process.
