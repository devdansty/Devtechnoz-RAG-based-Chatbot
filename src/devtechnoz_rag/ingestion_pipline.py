"""Load the DevTechNoz knowledge base and store it in Pinecone."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, SchemaBuilder

from .paths import DATA_DIR, PROJECT_ROOT

DEFAULT_SOURCE = DATA_DIR / "devtechnoz_chatbot_knowledge_base.txt"
CHUNK_SIZE = 500
BATCH_SIZE = 96
DEFAULT_EMBEDDING_MODEL = "llama-text-embed-v2"
TEXT_FIELD = "chunk_text"


def ingest_documents(
    source_path: Path,
    index_name: str,
    namespace: str,
    embedding_model: str,
    cloud: str,
    region: str,
) -> int:
    """Split a text file and let Pinecone embed and store its chunks."""
    if not source_path.is_file():
        raise FileNotFoundError(f"Source document was not found: {source_path}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=0,
        length_function=len,
        add_start_index=True,
    )
    text = source_path.read_text(encoding="utf-8")
    source = str(source_path.resolve())
    if not text.strip():
        raise ValueError(f"Source document is empty: {source_path}")

    documents = [Document(page_content=text, metadata={"source": source})]
    chunks = splitter.split_documents(documents)

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
            "_id": f"chunk-{chunk.metadata['start_index']}",
            TEXT_FIELD: chunk.page_content,
            "source": source,
            "start_index": chunk.metadata["start_index"],
        }
        for chunk in chunks
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

    return len(chunks)


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
        f"Successfully stored {chunk_count} chunks in Pinecone index "
        f"'{args.index_name}' (namespace '{args.namespace}')."
    )


if __name__ == "__main__":
    main()