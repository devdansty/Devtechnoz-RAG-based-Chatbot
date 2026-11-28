"""Persist website chat conversations in PostgreSQL."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator
from uuid import UUID

import psycopg


class ChatHistoryStorageError(RuntimeError):
    """Raised when website chat history cannot be read or written."""


@contextmanager
def _connection(database_url: str) -> Iterator[Any]:
    try:
        with psycopg.connect(database_url) as connection:
            yield connection
    except psycopg.Error as error:
        raise ChatHistoryStorageError(
            "The chat history database operation failed."
        ) from error


def initialize_database(database_url: str) -> None:
    """Create the conversation tables and lookup index when they are missing."""
    with _connection(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_conversations (
                    id UUID PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    id BIGSERIAL PRIMARY KEY,
                    conversation_id UUID NOT NULL
                        REFERENCES chat_conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK (role IN ('human', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS chat_messages_conversation_id_idx
                ON chat_messages (conversation_id, id)
                """
            )


def save_exchange(
    database_url: str,
    conversation_id: UUID,
    question: str,
    answer: str,
) -> None:
    """Store one completed visitor and assistant exchange atomically."""
    with _connection(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO chat_conversations (id)
                VALUES (%s)
                ON CONFLICT (id)
                DO UPDATE SET updated_at = CURRENT_TIMESTAMP
                """,
                (conversation_id,),
            )
            cursor.executemany(
                """
                INSERT INTO chat_messages (conversation_id, role, content)
                VALUES (%s, %s, %s)
                """,
                [
                    (conversation_id, "human", question),
                    (conversation_id, "assistant", answer),
                ],
            )


def load_recent_messages(
    database_url: str,
    conversation_id: UUID,
    limit: int,
) -> list[dict[str, str]]:
    """Load the newest messages in chronological order for chat restoration."""
    with _connection(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT role, content
                FROM chat_messages
                WHERE conversation_id = %s
                ORDER BY id DESC
                LIMIT %s
                """,
                (conversation_id, limit),
            )
            messages = cursor.fetchall()
    return [
        {"role": role, "content": content}
        for role, content in reversed(messages)
    ]


def list_conversations(
    database_url: str,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return recent conversation summaries for the protected admin page."""
    with _connection(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    conversation.id::text,
                    conversation.created_at,
                    conversation.updated_at,
                    COALESCE(
                        (
                            SELECT message.content
                            FROM chat_messages AS message
                            WHERE message.conversation_id = conversation.id
                                AND message.role = 'human'
                            ORDER BY message.id DESC
                            LIMIT 1
                        ),
                        ''
                    ) AS latest_question,
                    (
                        SELECT COUNT(*)
                        FROM chat_messages AS message
                        WHERE message.conversation_id = conversation.id
                    ) AS message_count
                FROM chat_conversations AS conversation
                ORDER BY conversation.updated_at DESC
                LIMIT %s
                OFFSET %s
                """,
                (limit, offset),
            )
            rows = cursor.fetchall()
    return [
        {
            "id": conversation_id,
            "created_at": created_at.isoformat(),
            "updated_at": updated_at.isoformat(),
            "latest_question": latest_question,
            "message_count": message_count,
        }
        for (
            conversation_id,
            created_at,
            updated_at,
            latest_question,
            message_count,
        ) in rows
    ]


def load_messages(
    database_url: str,
    conversation_id: UUID,
) -> list[dict[str, str]]:
    """Return every message in a conversation for transcript review."""
    with _connection(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT role, content, created_at
                FROM chat_messages
                WHERE conversation_id = %s
                ORDER BY id
                """,
                (conversation_id,),
            )
            rows = cursor.fetchall()
    return [
        {
            "role": role,
            "content": content,
            "created_at": created_at.isoformat(),
        }
        for role, content, created_at in rows
    ]


def delete_conversation(database_url: str, conversation_id: UUID) -> bool:
    """Delete one conversation and all its messages."""
    with _connection(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM chat_conversations WHERE id = %s",
                (conversation_id,),
            )
            return cursor.rowcount > 0


def delete_all_conversations(database_url: str) -> int:
    """Delete all conversations and return the number removed."""
    with _connection(database_url) as connection:
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM chat_conversations")
            return cursor.rowcount
