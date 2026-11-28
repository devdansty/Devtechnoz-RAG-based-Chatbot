"""Tests for the local web chat streaming endpoint."""

from __future__ import annotations

import base64
import json
import os
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch
from uuid import uuid4

from devtechnoz_rag.web_frontend import ChatRequestHandler, main


class WebFrontendTests(unittest.TestCase):
    def _start_server(self) -> str:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ChatRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        self.addCleanup(thread.join, timeout=5)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return f"http://127.0.0.1:{server.server_port}"

    def test_server_uses_platform_port_environment_variable(self) -> None:
        with (
            patch.dict(os.environ, {"PORT": "9123", "DATABASE_URL": ""}),
            patch("devtechnoz_rag.web_frontend.load_dotenv"),
            patch("devtechnoz_rag.web_frontend.ThreadingHTTPServer") as server,
            patch.object(sys, "argv", ["web_frontend"]),
        ):
            server.return_value.serve_forever.side_effect = KeyboardInterrupt

            main()

        server.assert_called_once_with(("127.0.0.1", 9123), ChatRequestHandler)

    def test_chat_endpoint_flushes_each_generated_chunk_as_sse(self) -> None:
        server_url = self._start_server()
        with patch.dict(os.environ, {"DATABASE_URL": ""}):
            request = Request(
                f"{server_url}/api/chat",
                data=json.dumps(
                    {
                        "question": "Hello",
                        "history": [["human", "Earlier question"]],
                    }
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with patch(
                "devtechnoz_rag.web_frontend.stream_answer",
                return_value=iter(["Hello", " there"]),
            ) as stream_answer:
                with urlopen(request, timeout=5) as response:
                    body = response.read().decode("utf-8")

            stream_answer.assert_called_once_with(
                "Hello",
                history=[("human", "Earlier question")],
            )
            self.assertEqual(
                response.headers.get_content_type(),
                "text/event-stream",
            )
            self.assertIn('event: message\ndata: {"text": "Hello"}', body)
            self.assertIn('event: message\ndata: {"text": " there"}', body)
            self.assertTrue(body.endswith("event: done\ndata: {}\n\n"))

    def test_completed_chat_exchange_is_saved_to_postgres(self) -> None:
        server_url = self._start_server()
        conversation_id = uuid4()
        request = Request(
            f"{server_url}/api/chat",
            data=json.dumps(
                {
                    "conversation_id": str(conversation_id),
                    "question": "Hello",
                    "history": [],
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with (
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}),
            patch(
                "devtechnoz_rag.web_frontend.stream_answer",
                return_value=iter(["Hi", " there"]),
            ),
            patch("devtechnoz_rag.web_frontend.save_exchange") as save_exchange,
        ):
            with urlopen(request, timeout=5) as response:
                body = response.read().decode("utf-8")

        self.assertTrue(body.endswith("event: done\ndata: {}\n\n"))
        save_exchange.assert_called_once_with(
            "postgresql://test",
            conversation_id,
            "Hello",
            "Hi there",
        )

    def test_chat_requires_conversation_id_when_persistence_is_enabled(self) -> None:
        server_url = self._start_server()
        request = Request(
            f"{server_url}/api/chat",
            data=json.dumps({"question": "Hello", "history": []}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}):
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=5)

        self.assertEqual(raised.exception.code, 400)

    def test_chat_history_restores_messages_using_conversation_header(self) -> None:
        server_url = self._start_server()
        conversation_id = uuid4()
        request = Request(
            f"{server_url}/api/chat/history",
            headers={"X-Conversation-ID": str(conversation_id)},
        )
        with (
            patch.dict(os.environ, {"DATABASE_URL": "postgresql://test"}),
            patch(
                "devtechnoz_rag.web_frontend.load_recent_messages",
                return_value=[{"role": "human", "content": "Hello"}],
            ) as load_recent,
        ):
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))

        self.assertTrue(payload["enabled"])
        self.assertEqual(payload["messages"], [{"role": "human", "content": "Hello"}])
        load_recent.assert_called_once_with(
            "postgresql://test",
            conversation_id,
            20,
        )

    def test_admin_conversation_list_requires_password(self) -> None:
        server_url = self._start_server()
        request = Request(f"{server_url}/api/admin/conversations")
        with (
            patch.dict(
                os.environ,
                {"DATABASE_URL": "postgresql://test", "CHAT_ADMIN_PASSWORD": "secret"},
            ),
            patch("devtechnoz_rag.web_frontend.list_conversations") as list_saved,
        ):
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=5)

        self.assertEqual(raised.exception.code, 401)
        list_saved.assert_not_called()

    def test_admin_page_is_served_at_admin_route(self) -> None:
        server_url = self._start_server()
        with urlopen(f"{server_url}/admin", timeout=5) as response:
            page = response.read().decode("utf-8")

        self.assertEqual(response.status, 200)
        self.assertIn("Conversation history", page)
        self.assertIn("/admin.js", page)

    def test_admin_conversation_list_accepts_configured_password(self) -> None:
        server_url = self._start_server()
        credentials = base64.b64encode(b"admin:secret").decode("ascii")
        request = Request(
            f"{server_url}/api/admin/conversations",
            headers={"Authorization": f"Basic {credentials}"},
        )
        with (
            patch.dict(
                os.environ,
                {"DATABASE_URL": "postgresql://test", "CHAT_ADMIN_PASSWORD": "secret"},
            ),
            patch(
                "devtechnoz_rag.web_frontend.list_conversations",
                return_value=[],
            ) as list_saved,
        ):
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload, {"conversations": []})
        list_saved.assert_called_once_with(
            "postgresql://test",
            limit=100,
            offset=0,
        )

    def test_admin_can_delete_one_conversation(self) -> None:
        server_url = self._start_server()
        conversation_id = uuid4()
        credentials = base64.b64encode(b"admin:secret").decode("ascii")
        request = Request(
            f"{server_url}/api/admin/conversation",
            headers={
                "Authorization": f"Basic {credentials}",
                "X-Conversation-ID": str(conversation_id),
            },
            method="DELETE",
        )
        with (
            patch.dict(
                os.environ,
                {"DATABASE_URL": "postgresql://test", "CHAT_ADMIN_PASSWORD": "secret"},
            ),
            patch(
                "devtechnoz_rag.web_frontend.delete_conversation",
                return_value=True,
            ) as delete_saved,
        ):
            with urlopen(request, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))

        self.assertEqual(payload, {"deleted": True})
        delete_saved.assert_called_once_with("postgresql://test", conversation_id)

    def test_history_rejects_non_string_roles(self) -> None:
        history, error = ChatRequestHandler._parse_history([[[], "invalid"]])

        self.assertEqual(history, [])
        self.assertEqual(error, "Conversation history contains an invalid message.")


if __name__ == "__main__":
    unittest.main()
