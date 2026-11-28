"""Tests for structured knowledge-base parsing and Pinecone ingestion."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from devtechnoz_rag.ingestion_pipline import (
    CHUNK_SIZE,
    _parse_documents,
    ingest_documents,
)


class IngestionPipelineTests(unittest.TestCase):
    def test_parser_creates_one_standalone_document_per_chunk_header(self) -> None:
        source_text = """Editorial notes that should not be indexed.
=== CHUNK: A01 | TITLE: Company | CATEGORY: company | STATUS: VERIFIED ===
Company information.
####################
PART B — DRAFT CONTENT
####################
=== CHUNK: B01 | TITLE: Project | CATEGORY: projects | STATUS: DRAFT ===
Project information.
"""

        documents = _parse_documents(source_text, "knowledge.txt")

        self.assertEqual(len(documents), 2)
        self.assertTrue(documents[0].page_content.startswith("=== CHUNK: A01"))
        self.assertNotIn("Editorial notes", documents[0].page_content)
        self.assertNotIn("PART B", documents[0].page_content)
        self.assertEqual(
            documents[0].metadata,
            {
                "source": "knowledge.txt",
                "start_index": source_text.index("=== CHUNK: A01"),
                "document_id": "A01",
                "title": "Company",
                "category": "company",
                "status": "VERIFIED",
            },
        )
        self.assertEqual(documents[1].metadata["status"], "DRAFT")

    def test_parser_rejects_missing_malformed_and_empty_documents(self) -> None:
        with self.assertRaisesRegex(ValueError, "no structured CHUNK"):
            _parse_documents("Plain text without document markers.", "source.txt")
        with self.assertRaisesRegex(ValueError, "malformed CHUNK"):
            _parse_documents("=== CHUNK: incomplete ===", "source.txt")
        with self.assertRaisesRegex(ValueError, "is empty"):
            _parse_documents(
                "=== CHUNK: A01 | TITLE: Empty | CATEGORY: test | STATUS: DRAFT ===",
                "source.txt",
            )

    def test_parser_requires_documents_to_fit_the_chunk_size(self) -> None:
        body = "x" * CHUNK_SIZE
        source_text = (
            "=== CHUNK: A01 | TITLE: Large | CATEGORY: test | STATUS: DRAFT ===\n"
            + body
        )

        with self.assertRaisesRegex(ValueError, "Split it into smaller"):
            _parse_documents(source_text, "source.txt")

    def test_ingestion_upserts_document_metadata_and_cleans_stale_records(self) -> None:
        source_text = """=== CHUNK: A01 | TITLE: Company | CATEGORY: company | STATUS: VERIFIED ===
Company information.
=== CHUNK: A02 | TITLE: Contact | CATEGORY: contact | STATUS: VERIFIED ===
Contact information.
"""
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "knowledge.txt"
            source_path.write_text(source_text, encoding="utf-8")

            documents_api = SimpleNamespace(
                upsert=Mock(),
                list=Mock(return_value=[SimpleNamespace(id="chunk-stale")]),
                fetch=Mock(
                    return_value=SimpleNamespace(
                        documents={
                            "chunk-stale": {"source": str(source_path.resolve())}
                        }
                    )
                ),
                delete=Mock(),
            )
            index = SimpleNamespace(documents=documents_api)
            pinecone = Mock()
            pinecone.has_index.return_value = True
            pinecone.describe_index.return_value = SimpleNamespace(host="index-host")
            pinecone.index.return_value = index

            with (
                patch.dict(os.environ, {"PINECONE_API_KEY": "test-key"}),
                patch("devtechnoz_rag.ingestion_pipline.Pinecone", return_value=pinecone),
            ):
                count = ingest_documents(
                    source_path=source_path,
                    index_name="test-index",
                    namespace="test-namespace",
                    embedding_model="test-model",
                    cloud="aws",
                    region="us-east-1",
                )

        records = documents_api.upsert.call_args.kwargs["documents"]
        self.assertEqual(count, 2)
        self.assertEqual([record["document_id"] for record in records], ["A01", "A02"])
        self.assertEqual(records[0]["category"], "company")
        self.assertEqual(records[0]["status"], "VERIFIED")
        self.assertIn("=== CHUNK: A01", records[0]["chunk_text"])
        documents_api.delete.assert_called_once_with(
            namespace="test-namespace",
            ids=["chunk-stale"],
        )


if __name__ == "__main__":
    unittest.main()
