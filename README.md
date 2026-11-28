# Devtechnoz-RAG-based-Chatbot

A Pinecone-backed retrieval-augmented chat application with a Groq-powered
terminal interface, streaming web chat, and optional PostgreSQL transcript
storage.

Try the deployed chatbot at [devtechnoz-chatbot.up.railway.app](https://devtechnoz-chatbot.up.railway.app).

## Project layout

| File or folder | Purpose |
|---|---|
| [`data/`](./data/) | Company prompt and source knowledge used by the RAG pipeline. |
| [`data/devtechnoz_base_prompt.txt`](./data/devtechnoz_base_prompt.txt) | System instructions that guide the assistant's tone and answers. |
| [`data/devtechnoz_chatbot_knowledge_base.txt`](./data/devtechnoz_chatbot_knowledge_base.txt) | Company facts ingested into Pinecone for retrieval. |
| [`docs/`](./docs/) | Project-specific guides. |
| [`docs/FRONTEND_BACKEND_FLOW.md`](./docs/FRONTEND_BACKEND_FLOW.md) | Explains browser requests, the Python server, Pinecone retrieval, and Groq streaming. |
| [`src/`](./src/) | Python source package root. |
| [`src/devtechnoz_rag/`](./src/devtechnoz_rag/) | Application modules. |
| [`src/devtechnoz_rag/__init__.py`](./src/devtechnoz_rag/__init__.py) | Marks the application package. |
| [`src/devtechnoz_rag/paths.py`](./src/devtechnoz_rag/paths.py) | Defines shared project, data, history, and web paths. |
| [`src/devtechnoz_rag/conversation_history.py`](./src/devtechnoz_rag/conversation_history.py) | Saves and loads terminal chat sessions as JSON. |
| [`src/devtechnoz_rag/ingestion_pipline.py`](./src/devtechnoz_rag/ingestion_pipline.py) | Splits the knowledge base and uploads it to Pinecone. |
| [`src/devtechnoz_rag/retrieval_pipline.py`](./src/devtechnoz_rag/retrieval_pipline.py) | Searches Pinecone and ranks diverse results with MMR. |
| [`src/devtechnoz_rag/rag_chat.py`](./src/devtechnoz_rag/rag_chat.py) | Builds the RAG prompt and streams Groq answers for terminal chat. |
| [`src/devtechnoz_rag/web_frontend.py`](./src/devtechnoz_rag/web_frontend.py) | Serves the website, streaming chat API, and protected transcript admin API. |
| [`src/devtechnoz_rag/chat_history_store.py`](./src/devtechnoz_rag/chat_history_store.py) | Persists website conversations in PostgreSQL. |
| [`tests/`](./tests/) | Automated unit and HTTP endpoint tests. |
| [`tests/test_conversation_history.py`](./tests/test_conversation_history.py) | Covers session persistence and history-aware answers. |
| [`tests/test_retrieval_pipline.py`](./tests/test_retrieval_pipline.py) | Covers Pinecone result selection and MMR diversity. |
| [`tests/test_web_frontend.py`](./tests/test_web_frontend.py) | Covers the streaming endpoint and server configuration. |
| [`web/`](./web/) | Static test frontend served by the Python web module. |
| [`web/index.html`](./web/index.html) | Page structure and chat interface elements. |
| [`web/styles.css`](./web/styles.css) | Responsive layout and visual styling. |
| [`web/app.js`](./web/app.js) | Handles chat input, sends requests, and displays streamed replies. |
| [`web/admin.html`](./web/admin.html) | Password-gated interface for reviewing and deleting chat transcripts. |
| [`web/admin.js`](./web/admin.js) and [`web/admin.css`](./web/admin.css) | Admin transcript page behavior and styling. |
| [`.env.example`](./.env.example) | Safe template for required and optional environment variables. |
| [`.gitignore`](./.gitignore) | Excludes credentials, virtual environments, caches, and local chat history. |
| [`pyproject.toml`](./pyproject.toml) | Package metadata, dependencies, and CLI entry points. |
| [`README.md`](./README.md) | Setup, run, test, and preview deployment instructions. |

The actual `.env`, virtual environment, Python caches, and generated
`data/conversations/` history files are local-only and intentionally excluded
from Git.

## Setup

Create `.env` from `.env.example`, add your Pinecone and Groq credentials, then
install the project into the active virtual environment:

```powershell
python -m pip install -e .
```

The editable install exposes the `devtechnoz_rag` package and its CLI commands.

## Run

Ingest the knowledge base:

```powershell
python -m devtechnoz_rag.ingestion_pipline
```

Start the terminal chat:

```powershell
python -m devtechnoz_rag.rag_chat
```

Start the test frontend and floating streaming chat:

```powershell
python -m devtechnoz_rag.web_frontend
```

Then open `http://127.0.0.1:8000`. The local web server binds to loopback by
default; API credentials stay server-side. This standalone page is only for
testing the chatbot and streaming flow, not the final DevTechNoz website.
The chatbot is intended to be integrated into the official DevTechNoz site.

## Persistent website chat history

Website transcripts are optional locally and use PostgreSQL in production.
For Railway, attach a PostgreSQL service to the web service and add a
`DATABASE_URL` variable that references the PostgreSQL service's connection
URL. Set `CHAT_ADMIN_PASSWORD` to a strong, unique secret in the web service's
Variables settings; do not put the real password in `.env.example` or commit it.
The application creates its required tables when it starts.

Open [the deployed admin page](https://devtechnoz-chatbot.up.railway.app/admin)
and enter that password to view full transcripts, delete individual
conversations, or delete all conversations. For another deployment, use
`https://<your-domain>/admin`.
Completed question-and-answer exchanges are kept until you delete them. The
website restores the latest 20 messages for a visitor when the page is reopened.
Without `DATABASE_URL`, chat still works but website transcripts are not saved.

## Optional free preview deployment

You can deploy the test frontend to Render to preview or share it during
development. This is not the final production deployment. The intended final
experience is to integrate the chatbot into the official DevTechNoz website.
The test web module serves both its temporary landing page and `/api/chat`;
you do not need a separate frontend service for this preview.

For a request-by-request explanation of how the browser, Python server,
Pinecone, and Groq interact, see
[the frontend/backend flow guide](./docs/FRONTEND_BACKEND_FLOW.md).

To deploy:

1. Push this project to a GitHub repository. Keep `.env` out of Git; it is
   ignored by `.gitignore`.
2. In Render, create a **Web Service** from that repository and select the free
   instance.
3. Use these settings:

   - **Build command:** `pip install -e .`
   - **Start command:** `python -m devtechnoz_rag.web_frontend --host 0.0.0.0`
   - **Health check path:** `/api/health`

   The server reads Render's `PORT` environment variable automatically.
4. In the Render service's **Environment** settings, add `PINECONE_API_KEY`,
   `PINECONE_INDEX_NAME`, and `GROQ_API_KEY`. Optionally add
   `PINECONE_NAMESPACE`, `GROQ_MODEL`, and `RAG_MAX_OUTPUT_TOKENS`.
5. Deploy and open the `onrender.com` preview URL Render provides.

If the Pinecone index is not populated yet, run
`python -m devtechnoz_rag.ingestion_pipline` locally first with the required
Pinecone settings. Do not commit `.env` or paste API keys into source files.

Render's free service spins down after 15 minutes without traffic, so its first
request after idle can take about a minute. Its filesystem is temporary, so
don't rely on local files for persistent deployed data. This public chat has no
authentication or rate limit; anyone who finds its preview URL can use your
Groq and Pinecone quotas. Keep it for testing unless you add usage protection
and complete the official-site integration.

## Test

```powershell
python -m unittest discover -s tests -v
```

