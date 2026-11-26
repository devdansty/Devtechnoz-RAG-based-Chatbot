"""Serve the DevTechNoz landing page and its streaming RAG chat endpoint."""

from __future__ import annotations

import argparse
import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv
from groq import GroqError
from pinecone import PineconeException

from .paths import PROJECT_ROOT, WEB_DIR
from .rag_chat import stream_answer


MAX_REQUEST_BYTES = 32_000
MAX_QUESTION_LENGTH = 2_000
MAX_HISTORY_MESSAGES = 20
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
        if urlsplit(self.path).path == "/api/health":
            self._send_json(200, {"status": "ok"})
            return
        super().do_GET()

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

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.close_connection = True

        try:
            for text in stream_answer(question, history=history):
                self._send_event("message", {"text": text})
            self._send_event("done", {})
        except (
            FileNotFoundError,
            GroqError,
            PineconeException,
            ValueError,
            RuntimeError,
        ) as error:
            self._send_event("error", {"message": str(error)})

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

    def _send_json(self, status: int, payload: dict[str, str]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_event(self, event: str, payload: dict[str, str]) -> None:
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
