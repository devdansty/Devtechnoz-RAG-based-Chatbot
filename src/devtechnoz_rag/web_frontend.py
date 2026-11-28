"""Serve the DevTechNoz landing page and its streaming RAG chat endpoint."""

from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import UUID

from dotenv import load_dotenv
from groq import GroqError
from pinecone import PineconeException

from .chat_history_store import (
    ChatHistoryStorageError,
    delete_all_conversations,
    delete_conversation,
    initialize_database,
    list_conversations,
    load_messages,
    load_recent_messages,
    save_exchange,
)
from .paths import PROJECT_ROOT, WEB_DIR
from .rag_chat import stream_answer


MAX_REQUEST_BYTES = 32_000
MAX_QUESTION_LENGTH = 2_000
MAX_HISTORY_MESSAGES = 20
ADMIN_CONVERSATIONS_PAGE_SIZE = 100
CONTENT_SECURITY_POLICY = (
    "default-src 'self'; style-src 'self'; script-src 'self'; "
    "connect-src 'self'; img-src 'self' data:; font-src 'self'; "
    "object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
)


class ChatRequestHandler(SimpleHTTPRequestHandler):
    """Serve static frontend assets and stream chat responses as SSE."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self) -> None:
        parsed_url = urlsplit(self.path)
        if parsed_url.path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        if parsed_url.path == "/admin":
            self.path = "/admin.html"
            super().do_GET()
            return
        if parsed_url.path == "/api/chat/history":
            self._get_chat_history()
            return
        if parsed_url.path == "/api/admin/conversations":
            if not self._require_admin():
                return
            query = parse_qs(parsed_url.query)
            try:
                offset = int(query.get("offset", ["0"])[0])
            except ValueError:
                self._send_json(400, {"error": "Invalid conversation list offset."})
                return
            if offset < 0 or offset > 1_000_000:
                self._send_json(400, {"error": "Invalid conversation list offset."})
                return
            database_url = self._database_url()
            if not database_url:
                self._send_json(503, {"error": "Chat history storage is not configured."})
                return
            try:
                self._send_json(
                    200,
                    {
                        "conversations": list_conversations(
                            database_url,
                            limit=ADMIN_CONVERSATIONS_PAGE_SIZE,
                            offset=offset,
                        )
                    },
                )
            except ChatHistoryStorageError as error:
                self._send_json(503, {"error": str(error)})
            return
        if parsed_url.path == "/api/admin/conversation":
            if not self._require_admin():
                return
            database_url = self._database_url()
            conversation_id = self._conversation_id(
                self.headers.get("X-Conversation-ID")
            )
            if not conversation_id:
                self._send_json(400, {"error": "Invalid conversation ID."})
                return
            if not database_url:
                self._send_json(503, {"error": "Chat history storage is not configured."})
                return
            try:
                self._send_json(
                    200,
                    {"messages": load_messages(database_url, conversation_id)},
                )
            except ChatHistoryStorageError as error:
                self._send_json(503, {"error": str(error)})
            return
        super().do_GET()

    def do_DELETE(self) -> None:
        path = urlsplit(self.path).path
        if path not in {
            "/api/admin/conversations",
            "/api/admin/conversation",
        }:
            self._send_json(404, {"error": "Not found."})
            return
        if not self._require_admin():
            return
        database_url = self._database_url()
        if not database_url:
            self._send_json(503, {"error": "Chat history storage is not configured."})
            return

        try:
            if path == "/api/admin/conversations":
                deleted_count = delete_all_conversations(database_url)
                self._send_json(200, {"deleted_count": deleted_count})
                return

            conversation_id = self._conversation_id(
                self.headers.get("X-Conversation-ID")
            )
            if not conversation_id:
                self._send_json(400, {"error": "Invalid conversation ID."})
                return
            deleted = delete_conversation(database_url, conversation_id)
            self._send_json(200, {"deleted": deleted})
        except ChatHistoryStorageError as error:
            self._send_json(503, {"error": str(error)})

    def end_headers(self) -> None:
        self.send_header("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        super().end_headers()

    def do_POST(self) -> None:
        if urlsplit(self.path).path != "/api/chat":
            self._send_json(404, {"error": "Not found."})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_json(400, {"error": "Invalid request size."})
            return
        if content_length < 1:
            self._send_json(400, {"error": "Request body is required."})
            return
        if content_length > MAX_REQUEST_BYTES:
            self._send_json(413, {"error": "Request is too large."})
            return

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json(400, {"error": "Request body must be valid JSON."})
            return
        if not isinstance(payload, dict):
            self._send_json(400, {"error": "Request body must be an object."})
            return

        question = payload.get("question")
        if not isinstance(question, str) or not question.strip():
            self._send_json(400, {"error": "Enter a message before sending."})
            return
        question = question.strip()
        if len(question) > MAX_QUESTION_LENGTH:
            self._send_json(
                413,
                {"error": f"Messages must be {MAX_QUESTION_LENGTH} characters or fewer."},
            )
            return

        history, history_error = self._parse_history(payload.get("history", []))
        if history_error is not None:
            self._send_json(400, {"error": history_error})
            return

        database_url = self._database_url()
        raw_conversation_id = payload.get("conversation_id")
        conversation_id = None
        if raw_conversation_id is not None:
            conversation_id = self._conversation_id(raw_conversation_id)
            if not conversation_id:
                self._send_json(400, {"error": "Invalid conversation ID."})
                return
        if database_url and conversation_id is None:
            self._send_json(400, {"error": "A valid conversation ID is required."})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.close_connection = True

        try:
            answer_parts = []
            for text in stream_answer(question, history=history):
                answer_parts.append(text)
                self._send_event("message", {"text": text})
            if database_url and conversation_id:
                save_exchange(
                    database_url,
                    conversation_id,
                    question,
                    "".join(answer_parts),
                )
            self._send_event("done", {})
        except (
            ChatHistoryStorageError,
            FileNotFoundError,
            GroqError,
            PineconeException,
            ValueError,
            RuntimeError,
        ) as error:
            self._send_event("error", {"message": str(error)})

    def _get_chat_history(self) -> None:
        conversation_id = self._conversation_id(
            self.headers.get("X-Conversation-ID")
        )
        if not conversation_id:
            self._send_json(400, {"error": "A valid conversation ID is required."})
            return

        database_url = self._database_url()
        if not database_url:
            self._send_json(200, {"enabled": False, "messages": []})
            return
        try:
            messages = load_recent_messages(
                database_url,
                conversation_id,
                MAX_HISTORY_MESSAGES,
            )
            self._send_json(200, {"enabled": True, "messages": messages})
        except ChatHistoryStorageError as error:
            self._send_json(503, {"error": str(error)})

    def _require_admin(self) -> bool:
        expected_password = os.environ.get("CHAT_ADMIN_PASSWORD")
        if not expected_password:
            self._send_json(503, {"error": "Chat admin access is not configured."})
            return False

        authorization = self.headers.get("Authorization", "")
        if not authorization.startswith("Basic "):
            self._send_json(401, {"error": "Enter the admin password to continue."})
            return False
        try:
            credentials = base64.b64decode(
                authorization[6:],
                validate=True,
            ).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            self._send_json(401, {"error": "Invalid admin credentials."})
            return False

        username, separator, password = credentials.partition(":")
        if (
            not separator
            or username != "admin"
            or not hmac.compare_digest(password, expected_password)
        ):
            self._send_json(401, {"error": "Invalid admin credentials."})
            return False
        return True

    @staticmethod
    def _database_url() -> str | None:
        return os.environ.get("DATABASE_URL") or None

    @staticmethod
    def _conversation_id(value: object) -> UUID | None:
        if not isinstance(value, str):
            return None
        try:
            return UUID(value)
        except ValueError:
            return None

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[web] {self.address_string()} - {format % args}")

    @staticmethod
    def _parse_history(
        value: object,
    ) -> tuple[list[tuple[str, str]], str | None]:
        if not isinstance(value, list):
            return [], "Conversation history must be a list."

        messages: list[tuple[str, str]] = []
        for item in value[-MAX_HISTORY_MESSAGES:]:
            if not isinstance(item, list) or len(item) != 2:
                return [], "Conversation history contains an invalid message."
            role, content = item
            if (
                not isinstance(role, str)
                or role not in {"human", "assistant"}
                or not isinstance(content, str)
            ):
                return [], "Conversation history contains an invalid message."
            messages.append((role, content))
        return messages, None

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_event(self, event: str, payload: dict[str, Any]) -> None:
        body = (
            f"event: {event}\n"
            f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        ).encode("utf-8")
        self.wfile.write(body)
        self.wfile.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the DevTechNoz website and streaming RAG chat."
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Interface to bind to (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("PORT", "8000")),
        help="Port to listen on (default: 8000).",
    )
    args = parser.parse_args()

    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        initialize_database(database_url)
        print("[web] PostgreSQL chat history is enabled.")
        if not os.environ.get("CHAT_ADMIN_PASSWORD"):
            print("[web] CHAT_ADMIN_PASSWORD is missing; the admin page is disabled.")
    else:
        print("[web] DATABASE_URL is missing; website chat history is not persisted.")
    server = ThreadingHTTPServer((args.host, args.port), ChatRequestHandler)
    print(f"DevTechNoz web chat is available at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the web chat.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
