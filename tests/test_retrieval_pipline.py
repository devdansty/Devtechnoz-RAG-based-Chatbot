"""Tests for Pinecone retrieval and maximal marginal relevance selection."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from devtechnoz_rag.retrieval_pipline import retrieve_chunks


class RetrievalPipelineTests(unittest.TestCase):
    def test_retrieve_chunks_overfetches_and_selects_diverse_results(self) -> None:
        candidates = [
            SimpleNamespace(
                id="first",
                score=0.90,
                chunk_text="cats love crunchy tuna food",
                source="source.txt",
                start_index=0,
            ),
            SimpleNamespace(
                id="similar",
                score=0.86,
                chunk_text="cats love crunchy fish food",
                source="source.txt",
                start_index=100,
            ),
            SimpleNamespace(
                id="diverse",
                score=0.70,
                chunk_text="quantum mechanics particle wave functions",
                source="source.txt",
                start_index=200,
            ),
        ]
        documents = SimpleNamespace(
            search=Mock(return_value=SimpleNamespace(matches=candidates))
        )
        index = SimpleNamespace(documents=documents)

        with (
            patch("devtechnoz_rag.retrieval_pipline.load_dotenv"),
            patch("devtechnoz_rag.retrieval_pipline._get_pinecone_index", return_value=index),
        ):
            chunks = retrieve_chunks("test prompt", top_k=2)

        self.assertEqual([chunk["id"] for chunk in chunks], ["first", "diverse"])
        documents.search.assert_called_once()
        self.assertEqual(documents.search.call_args.kwargs["top_k"], 8)


if __name__ == "__main__":
    unittest.main()
