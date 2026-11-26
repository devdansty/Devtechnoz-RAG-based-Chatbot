"""Tests for persisted conversation history and history-aware retrieval."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from devtechnoz_rag.conversation_history import (
    DEFAULT_HISTORY_PATH,
    ConversationHistory,
)
from devtechnoz_rag.paths import PROJECT_ROOT
from devtechnoz_rag.rag_chat import DEFAULT_MODEL, stream_answer


class ConversationHistoryTests(unittest.TestCase):
    def test_default_and_relative_history_paths_use_project_layout(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            default_history = ConversationHistory(session_id="default")
        with patch.dict(
            os.environ,
            {"RAG_HISTORY_FILE": "data/custom/history.json"},
            clear=True,
        ):
            configured_history = ConversationHistory(session_id="configured")

        self.assertEqual(default_history.base_path, DEFAULT_HISTORY_PATH)
        self.assertEqual(
            default_history.path.parent,
            DEFAULT_HISTORY_PATH.parent,
        )
        self.assertEqual(
            configured_history.base_path,
            PROJECT_ROOT / "data" / "custom" / "history.json",
        )

    def test_groq_client_uses_key_and_model_from_environment(self) -> None:
        response_chunk = SimpleNamespace(
            content="Answer",
            response_metadata={},
        )
        with (
            patch("devtechnoz_rag.rag_chat.load_dotenv"),
            patch("devtechnoz_rag.rag_chat.retrieve_chunks", return_value=[]),
            patch.dict(
                os.environ,
                {
                    "groq_api_key": "test-key",
                    "GROQ_MODEL": "test-model",
                },
            ),
            patch("devtechnoz_rag.rag_chat.ChatGroq") as chat_groq,
        ):
            chat_groq.return_value.stream.return_value = [response_chunk]

            answer = "".join(stream_answer("Tell me about the company"))

        self.assertEqual(answer, "Answer")
        call_kwargs = chat_groq.call_args.kwargs
        self.assertEqual(
            call_kwargs["api_key"].get_secret_value(),
            "test-key",
        )
        self.assertEqual(
            {key: value for key, value in call_kwargs.items() if key != "api_key"},
            {"model": "test-model", "max_tokens": 1200, "streaming": True},
        )

    def test_groq_client_uses_supported_default_model(self) -> None:
        response_chunk = SimpleNamespace(
            content="Answer",
            response_metadata={},
        )
        with (
            patch("devtechnoz_rag.rag_chat.load_dotenv"),
            patch("devtechnoz_rag.rag_chat.retrieve_chunks", return_value=[]),
            patch.dict(os.environ, {"groq_api_key": "test-key"}, clear=True),
            patch("devtechnoz_rag.rag_chat.ChatGroq") as chat_groq,
        ):
            chat_groq.return_value.stream.return_value = [response_chunk]

            answer = "".join(stream_answer("Tell me about the company"))

        self.assertEqual(answer, "Answer")
        self.assertEqual(chat_groq.call_args.kwargs["model"], DEFAULT_MODEL)

    def test_completed_exchanges_persist_and_reload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.json"
            history = ConversationHistory(history_path, session_id="test-session")

            history.add_exchange("Hello", "Hi there")

            loaded = ConversationHistory(
                history_path,
                session_id=history.session_id,
            )
            self.assertEqual(loaded.messages, [("human", "Hello"), ("assistant", "Hi there")])
            self.assertEqual(loaded.turn_count, 1)

    def test_each_session_gets_a_new_id_and_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.json"
            first = ConversationHistory(history_path)
            first.add_exchange("Hello", "Hi there")
            previous_id = first.session_id

            second = ConversationHistory(history_path)

            self.assertNotEqual(first.session_id, second.session_id)
            self.assertNotEqual(first.path, second.path)
            self.assertTrue(first.path.is_file())
            self.assertFalse(second.path.exists())

            prior_path = first.path
            new_id = first.start_new_session()
            self.assertNotEqual(new_id, previous_id)
            self.assertNotEqual(first.session_id, second.session_id)
            self.assertEqual(first.session_id, new_id)
            self.assertNotEqual(first.path, prior_path)
            self.assertEqual(first.messages, [])

    def test_invalid_history_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            history_path = Path(directory) / "history.json"
            session_id = "invalid-session"
            session_path = ConversationHistory(
                history_path,
                session_id=session_id,
            ).path
            session_path.write_text(
                json.dumps([{"role": "assistant", "content": "Hi"}])
            )

            with self.assertRaisesRegex(ValueError, "invalid message"):
                ConversationHistory(history_path, session_id=session_id)

    def test_recent_history_is_sent_to_model_and_retrieval(self) -> None:
        previous_history = [
            ("human", "What services do you offer?"),
            ("assistant", "We build websites."),
        ]
        model_messages: list[list[tuple[str, str]]] = []

        class FakeClient:
            def stream(
                self,
                messages: list[tuple[str, str]],
            ) -> list[SimpleNamespace]:
                model_messages.append(messages)
                return [SimpleNamespace(content="We also build apps.", response_metadata={})]

        with patch("devtechnoz_rag.rag_chat.retrieve_chunks", return_value=[]) as retrieve:
            self.assertEqual(
                "".join(
                    stream_answer(
                        "What about mobile?",
                        client=FakeClient(),
                        history=previous_history,
                    )
                ),
                "We also build apps.",
            )

        retrieve.assert_called_once_with(
            "What services do you offer?\nWhat about mobile?",
            5,
        )
        self.assertEqual(model_messages[0][1:3], previous_history)
        self.assertTrue(
            model_messages[0][-1][1].endswith(
                "Visitor question: What about mobile?"
            )
        )


if __name__ == "__main__":
    unittest.main()
