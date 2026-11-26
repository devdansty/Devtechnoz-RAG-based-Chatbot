"""Retrieve relevant text chunks from the Pinecone document index."""

from __future__ import annotations

import argparse
import math
import os
import re
from collections import Counter
from functools import lru_cache

from dotenv import load_dotenv
from pinecone import Pinecone

from .paths import PROJECT_ROOT

TEXT_FIELD = "chunk_text"
TEST_PROMPT = "What services does DevTechNoz offer?"
DEFAULT_TOP_K = 5
MMR_FETCH_MULTIPLIER = 4
MMR_LAMBDA = 0.5
TOKEN_PATTERN = re.compile(r"[a-z0-9]+", re.IGNORECASE)


def _chunk_score(chunk: dict[str, object]) -> float:
    score = chunk["score"]
    if not isinstance(score, (int, float)):
        raise TypeError("Pinecone returned a chunk without a numeric score.")
    return float(score)


def _chunk_text(chunk: dict[str, object]) -> str:
    text = chunk["text"]
    if not isinstance(text, str):
        raise TypeError("Pinecone returned a chunk without text.")
    return text


def _text_similarity(left: str, right: str) -> float:
    """Measure lexical cosine similarity for integrated-embedding results."""
    left_terms = Counter(TOKEN_PATTERN.findall(left.casefold()))
    right_terms = Counter(TOKEN_PATTERN.findall(right.casefold()))
    left_norm = math.sqrt(sum(count * count for count in left_terms.values()))
    right_norm = math.sqrt(sum(count * count for count in right_terms.values()))
    if not left_norm or not right_norm:
        return 0.0
    shared_terms = left_terms.keys() & right_terms.keys()
    dot_product = sum(left_terms[term] * right_terms[term] for term in shared_terms)
    return dot_product / (left_norm * right_norm)


def _select_mmr_chunks(
    chunks: list[dict[str, object]],
    top_k: int,
) -> list[dict[str, object]]:
    """Balance Pinecone relevance with lexical diversity using MMR."""
    selected: list[dict[str, object]] = []
    remaining = list(range(len(chunks)))
    while remaining and len(selected) < top_k:
        best_index = max(
            remaining,
            key=lambda index: MMR_LAMBDA * _chunk_score(chunks[index])
            - (1 - MMR_LAMBDA)
            * max(
                (
                    _text_similarity(
                        _chunk_text(chunks[index]),
                        _chunk_text(chosen),
                    )
                    for chosen in selected
                ),
                default=0.0,
            ),
        )
        selected.append(chunks[best_index])
        remaining.remove(best_index)
    return selected


@lru_cache(maxsize=1)
def _get_pinecone_index():
    """Create and reuse the configured Pinecone data-plane index client."""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    if not api_key:
        raise ValueError("Set PINECONE_API_KEY in the environment or .env file.")
    if not index_name:
        raise ValueError("Set PINECONE_INDEX_NAME in the environment or .env file.")

    pinecone = Pinecone(api_key=api_key)
    if not pinecone.has_index(index_name):
        raise ValueError(f"Pinecone index '{index_name}' does not exist.")

    index_host = pinecone.describe_index(index_name).host
    if not index_host:
        raise RuntimeError(f"Pinecone did not return a host for index '{index_name}'.")
    return pinecone.index(host=index_host)


def retrieve_chunks(prompt: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, object]]:
    """Retrieve relevant, diverse chunks using Pinecone search and MMR."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("The search prompt cannot be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    load_dotenv(PROJECT_ROOT / ".env")
    namespace = os.getenv("PINECONE_NAMESPACE", "devtechnoz")
    index = _get_pinecone_index()
    response = index.documents.search(
        namespace=namespace,
        top_k=top_k * MMR_FETCH_MULTIPLIER,
        score_by=[
            {
                "type": "embed",
                "fields": [TEXT_FIELD],
                "query": prompt,
            }
        ],
        include_fields=[TEXT_FIELD, "source", "start_index"],
    )

    candidates = [
        {
            "id": match.id,
            "score": match.score,
            "text": match.chunk_text,
            "source": match.source,
            "start_index": match.start_index,
        }
        for match in response.matches
    ]
    return _select_mmr_chunks(candidates, top_k)


def test_retrieval(
    prompt: str = TEST_PROMPT,
    top_k: int = DEFAULT_TOP_K,
) -> list[dict[str, object]]:
    """Print and return chunks matching the prompt, for manual retrieval checks."""
    chunks = retrieve_chunks(prompt, top_k)
    print(f"Prompt: {prompt}")
    if not chunks:
        print("No matching chunks found.")
        return chunks

    for rank, chunk in enumerate(chunks, start=1):
        print(
            f"\n[{rank}] score={chunk['score']} | id={chunk['id']} "
            f"| source={chunk['source']} | start_index={chunk['start_index']}"
        )
        print(chunk["text"])
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search the Pinecone index and print matching text chunks."
    )
    parser.add_argument(
        "--prompt",
        default=TEST_PROMPT,
        help="Text to search for (defaults to the TEST_PROMPT variable).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of matching chunks to return (default: {DEFAULT_TOP_K}).",
    )
    args = parser.parse_args()
    test_retrieval(args.prompt, args.top_k)


if __name__ == "__main__":
    main()