"""Load and persist the terminal chat's conversation history as JSON."""

from __future__ import annotations

import json
import os
from pathlib import Path
from uuid import uuid4

from .paths import CONVERSATION_DIR, PROJECT_ROOT

DEFAULT_HISTORY_PATH = CONVERSATION_DIR / "conversation_history.json"
VALID_ROLES = {"human", "assistant"}


class ConversationHistory:
    """Persist completed exchanges and expose them as model chat messages."""

    def __init__(
        self,
        path: Path | None = None,
        session_id: str | None = None,
    ) -> None:
        configured_path = os.getenv("RAG_HISTORY_FILE")
        if path is not None:
            self.base_path = path
        elif configured_path:
            configured = Path(configured_path)
            self.base_path = (
                configured if configured.is_absolute() else PROJECT_ROOT / configured
            )
        else:
            self.base_path = DEFAULT_HISTORY_PATH
        self.session_id = session_id or uuid4().hex
        self.path = self._session_path()
        self._messages = self._load()

    def start_new_session(self) -> str:
        """Create a fresh session ID and retain older sessions on disk."""
        self.session_id = uuid4().hex
        self.path = self._session_path()
        self._messages = []
        return self.session_id

    @property
    def messages(self) -> list[tuple[str, str]]:
        """Return a copy of the saved messages in model-compatible format."""
        return [
            (message["role"], message["content"])
            for message in self._messages
        ]

    @property
    def turn_count(self) -> int:
        """Return the number of complete user/assistant exchanges."""
        return len(self._messages) // 2

    def add_exchange(self, user_message: str, assistant_message: str) -> None:
        """Append one completed exchange and save it to disk."""
        if not user_message.strip() or not assistant_message.strip():
            raise ValueError("Both conversation messages must be non-empty.")

        updated_messages = [
            *self._messages,
            {"role": "human", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        self._save(updated_messages)
        self._messages = updated_messages

    def _load(self) -> list[dict[str, str]]:
        if not self.path.exists():
            return []

        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError(f"History file is not valid JSON: {self.path}") from error

        if not isinstance(data, list):
            raise ValueError(f"History file must contain a JSON list: {self.path}")

        messages: list[dict[str, str]] = []
        for item in data:
            role = item.get("role") if isinstance(item, dict) else None
            content = item.get("content") if isinstance(item, dict) else None
            expected_role = "human" if len(messages) % 2 == 0 else "assistant"
            if (
                not isinstance(role, str)
                or role not in VALID_ROLES
                or role != expected_role
                or not isinstance(content, str)
                or not content.strip()
            ):
                raise ValueError(f"History file contains an invalid message: {self.path}")
            messages.append({"role": role, "content": content})

        if len(messages) % 2:
            raise ValueError(f"History file contains an incomplete exchange: {self.path}")
        return messages

    def _session_path(self) -> Path:
        return self.base_path.with_name(
            f"{self.base_path.stem}_{self.session_id}{self.base_path.suffix}"
        )

    def _save(self, messages: list[dict[str, str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(messages, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
