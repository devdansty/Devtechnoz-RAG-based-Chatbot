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
  | POST /api/chat
  | JSON: question + recent conversation history
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
```

## What happens when you send a message

1. The page in [`index.html`](../web/index.html) loads the chat interface.
   It also loads [`app.js`](../web/app.js), which handles the launcher, input,
   conversation context, and streaming display.
2. On submit, the browser sends a same-origin `POST /api/chat` request. Its JSON
   body contains the new question and up to 20 recent messages from the current
   page session.
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

The page keeps its conversation context in browser memory. Refreshing the page
starts a new web-chat context; the web interface does not save that chat to the
server's conversation-history files.

## Useful routes

| Route | Purpose |
|---|---|
| `GET /` | Serves the landing page. |
| `GET /styles.css` and `GET /app.js` | Serves the frontend assets. |
| `GET /api/health` | Returns `{"status": "ok"}` for a deployment health check. |
| `POST /api/chat` | Accepts a question and streams the answer as SSE. |

## Where configuration lives

Locally, the Python backend reads API keys from the project-root `.env` file.
On a hosting provider, set the same values as service environment variables.
The browser never receives the Pinecone or Groq API keys.

The web server and the API use the same host and port. Locally, open
`http://127.0.0.1:8000`; on a hosted service, the platform's URL points to this
same Python process.
