
import os
import hashlib

from langchain_chroma import Chroma

from rag.embeddings import get_embeddings
from rag.ingestion import load_pdf, text_splitting


def get_vectorstore(
    persist_dir: str = "./chroma_db"
):
    embeddings = get_embeddings()

    return Chroma(
        collection_name="pdf_knowledge_base",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )


def ingest_pdfs_to_chroma(
    persist_dir: str,
    pdf_paths: list[str],
    batch_size: int = 64
) -> str:

    vectorstore = get_vectorstore(persist_dir)

    docs = load_pdf(pdf_paths)
    chunks = text_splitting(docs)

    if not chunks:
        return "No extractable text found in documents."

    total_chunks = len(chunks)

    # Process chunks in batches to reduce CPU/memory pressure.
    for i in range(0, total_chunks, batch_size):

        batch = chunks[i:i + batch_size]
        ids = []

        for chunk in batch:

            source = os.path.basename(
                chunk.metadata.get("source", "unknown")
            )

            page = chunk.metadata.get("page", 0)

            content_hash = hashlib.md5(
                chunk.page_content.strip()
                .encode("utf-8")
            ).hexdigest()[:8]

            ids.append(
                f"{source}_p{page}_{content_hash}"
            )

        vectorstore.add_documents(
            documents=batch,
            ids=ids
        )

    return (
        f"Indexed {total_chunks} chunk(s) "
        f"across {len(pdf_paths)} document(s)."
    )
