"""Tests for the local web chat streaming endpoint."""

from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.request import Request, urlopen
from unittest.mock import patch

from devtechnoz_rag.web_frontend import ChatRequestHandler, main


class WebFrontendTests(unittest.TestCase):
    def test_server_uses_platform_port_environment_variable(self) -> None:
        with (
            patch.dict(os.environ, {"PORT": "9123"}),
            patch("devtechnoz_rag.web_frontend.load_dotenv"),
            patch("devtechnoz_rag.web_frontend.ThreadingHTTPServer") as server,
            patch.object(sys, "argv", ["web_frontend"]),
        ):
            server.return_value.serve_forever.side_effect = KeyboardInterrupt

            main()

        server.assert_called_once_with(("127.0.0.1", 9123), ChatRequestHandler)

    def test_chat_endpoint_flushes_each_generated_chunk_as_sse(self) -> None:
        server = ThreadingHTTPServer(("127.0.0.1", 0), ChatRequestHandler)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{server.server_port}/api/chat",
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
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_history_rejects_non_string_roles(self) -> None:
        history, error = ChatRequestHandler._parse_history([[[], "invalid"]])

        self.assertEqual(history, [])
        self.assertEqual(error, "Conversation history contains an invalid message.")


if __name__ == "__main__":
    unittest.main()
