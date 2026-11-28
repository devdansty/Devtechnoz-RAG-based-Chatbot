"""Parse structured DevTechNoz knowledge documents and store them in Pinecone."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from pinecone import Pinecone, SchemaBuilder

from .paths import DATA_DIR, PROJECT_ROOT

DEFAULT_SOURCE = DATA_DIR / "devtechnoz_chatbot_knowledge_base.txt"
CHUNK_SIZE = 2_000
BATCH_SIZE = 96
DEFAULT_EMBEDDING_MODEL = "llama-text-embed-v2"
TEXT_FIELD = "chunk_text"

CHUNK_HEADER_PATTERN = re.compile(
    r"^=== CHUNK:\s*(?P<document_id>[^|]+?)\s*"
    r"\|\s*TITLE:\s*(?P<title>.*?)\s*"
    r"\|\s*CATEGORY:\s*(?P<category>.*?)\s*"
    r"\|\s*STATUS:\s*(?P<status>.*?)\s*===[ \t]*$",
    re.MULTILINE,
)
CHUNK_MARKER_PATTERN = re.compile(r"^=== CHUNK:", re.MULTILINE)
SECTION_LINE_PATTERN = re.compile(
    r"^(?:#{20,}|PART [A-Z]\b.*)$",
    re.MULTILINE,
)


def _parse_documents(text: str, source: str) -> list[Document]:
    """Parse each explicit knowledge-base chunk into a standalone document."""
    matches = list(CHUNK_HEADER_PATTERN.finditer(text))
    marker_count = len(CHUNK_MARKER_PATTERN.findall(text))
    if marker_count != len(matches):
        raise ValueError("The knowledge base contains a malformed CHUNK header.")
    if not matches:
        raise ValueError(
            "The knowledge base has no structured CHUNK headers to ingest."
        )

    documents: list[Document] = []
    for position, match in enumerate(matches):
        next_start = (
            matches[position + 1].start()
            if position + 1 < len(matches)
            else len(text)
        )
        body = text[match.end() : next_start]
        body = SECTION_LINE_PATTERN.sub("", body).strip()
        if not body:
            raise ValueError(
                f"Knowledge document '{match.group('document_id').strip()}' is empty."
            )

        header = match.group(0).strip()
        page_content = f"{header}\n{body}"
        if len(page_content) > CHUNK_SIZE:
            document_id = match.group("document_id").strip()
            raise ValueError(
                f"Knowledge document '{document_id}' is {len(page_content)} "
                f"characters; the maximum is {CHUNK_SIZE}. Split it into smaller "
                "self-contained CHUNK documents in the source file."
            )

        metadata = {
            "source": source,
            "start_index": match.start(),
            "document_id": match.group("document_id").strip(),
            "title": match.group("title").strip(),
            "category": match.group("category").strip(),
            "status": match.group("status").strip(),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def ingest_documents(
    source_path: Path,
    index_name: str,
    namespace: str,
    embedding_model: str,
    cloud: str,
    region: str,
) -> int:
    """Ingest structured documents, preserving their source-defined boundaries."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Source document was not found: {source_path}")

    text = source_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Source document is empty: {source_path}")
    source = str(source_path.resolve())
    documents = _parse_documents(text, source)

    pinecone = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    if pinecone.has_index(index_name):
        index_host = pinecone.describe_index(index_name).host
    else:
        schema = (
            SchemaBuilder()
            .add_string_field(
                name=TEXT_FIELD,
                embed={"model": embedding_model},
            )
            .build()
        )
        index_model = pinecone.indexes.create(
            name=index_name,
            schema=schema,
            deployment={
                "deployment_type": "managed",
                "cloud": cloud,
                "region": region,
            },
        )
        index_host = index_model.host
    if not index_host:
        raise RuntimeError(f"Pinecone did not return a host for index '{index_name}'.")
    index = pinecone.index(host=index_host)

    records = [
        {
            "_id": f"chunk-{document.metadata['start_index']}",
            TEXT_FIELD: document.page_content,
            "source": source,
            "start_index": document.metadata["start_index"],
            "document_id": document.metadata["document_id"],
            "title": document.metadata["title"],
            "category": document.metadata["category"],
            "status": document.metadata["status"],
        }
        for document in documents
    ]
    for batch_start in range(0, len(records), BATCH_SIZE):
        index.documents.upsert(
            namespace=namespace,
            documents=records[batch_start : batch_start + BATCH_SIZE],
        )

    current_ids = {record["_id"] for record in records}
    existing_ids = [
        document.id
        for document in index.documents.list(namespace=namespace, prefix="chunk-")
    ]
    stale_ids: list[str] = []
    for batch_start in range(0, len(existing_ids), BATCH_SIZE):
        response = index.documents.fetch(
            namespace=namespace,
            ids=existing_ids[batch_start : batch_start + BATCH_SIZE],
            include_fields=["source"],
        )
        stale_ids.extend(
            document_id
            for document_id, document in response.documents.items()
            if document.get("source") == source and document_id not in current_ids
        )
    for batch_start in range(0, len(stale_ids), BATCH_SIZE):
        index.documents.delete(
            namespace=namespace,
            ids=stale_ids[batch_start : batch_start + BATCH_SIZE],
        )

    return len(documents)


def main() -> None:
    load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(
        description="Embed the DevTechNoz knowledge base and store it in Pinecone."
    )
    parser.add_argument(
        "source",
        nargs="?",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"UTF-8 text document to ingest (default: {DEFAULT_SOURCE.name})",
    )
    parser.add_argument(
        "--index-name",
        default=os.getenv("PINECONE_INDEX_NAME"),
        help="Existing Pinecone index name (defaults to PINECONE_INDEX_NAME).",
    )
    parser.add_argument(
        "--namespace",
        default=os.getenv("PINECONE_NAMESPACE", "devtechnoz"),
        help="Pinecone namespace (defaults to PINECONE_NAMESPACE or 'devtechnoz').",
    )
    args = parser.parse_args()

    required_variables = ("PINECONE_API_KEY",)
    missing_variables = [
        variable for variable in required_variables if not os.getenv(variable)
    ]
    if missing_variables:
        parser.error(
            "Missing environment variables: "
            + ", ".join(missing_variables)
            + ". Set them in .env or your environment."
        )
    if not args.index_name:
        parser.error("Provide --index-name or set PINECONE_INDEX_NAME in .env.")

    try:
        chunk_count = ingest_documents(
            source_path=args.source,
            index_name=args.index_name,
            namespace=args.namespace,
            embedding_model=os.getenv(
                "PINECONE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            ),
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Successfully stored {chunk_count} documents in Pinecone index "
        f"'{args.index_name}' (namespace '{args.namespace}')."
    )


if __name__ == "__main__":
    main()
