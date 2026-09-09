import os
import hashlib
from pathlib import Path
from glob import glob

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.embeddings import get_embeddings


def get_vectorstore(persist_dir: str = "chroma_db"):
    return Chroma(
        collection_name="pdf_knowledge_base",
        embedding_function=get_embeddings(),
        persist_directory=persist_dir
    )


def load_and_split_pdfs(pdf_paths: list[str], chunk_size: int = 1500, chunk_overlap: int = 250) -> list:
    """Loads PDFs and splits them into chunks optimized for academic texts."""
    all_documents = []
    for path in pdf_paths:
        try:
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            all_documents.extend(docs)
            print(f"Loaded: {Path(path).name} ({len(docs)} pages)")
        except Exception as e:
            print(f"Error loading {path}: {e}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(all_documents)


def ingest_pdfs_to_chroma(
    persist_dir: str = "chroma_db",
    pdf_paths: list[str] = None,
    batch_size: int = 64
) -> str:
    """
    Idempotently indexes PDFs into ChromaDB with deterministic MD5 IDs.
    Compatible with both server.py calls and manual execution.
    """
    if pdf_paths is None:
        pdf_paths = glob(os.path.join("data", "*.pdf"))

    if not pdf_paths:
        return "No PDFs found in the data directory."

    chunks = load_and_split_pdfs(pdf_paths, chunk_size=1500, chunk_overlap=250)
    if not chunks:
        return "No extractable text found in documents."

    vectorstore = get_vectorstore(persist_dir)
    total_chunks = len(chunks)

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i + batch_size]
        ids = []
        for chunk in batch:
            source = os.path.basename(chunk.metadata.get("source", "unknown"))
            # 12-char deterministic MD5 hash matching build_ground_truth
            content_hash = hashlib.md5(
                chunk.page_content.strip().encode("utf-8")
            ).hexdigest()[:12]
            ids.append(f"{source}_{content_hash}")

        vectorstore.add_documents(documents=batch, ids=ids)

    status_msg = f"Indexed {total_chunks} chunk(s) across {len(pdf_paths)} document(s)."
    print(status_msg)
    return status_msg


if __name__ == "__main__":
    ingest_pdfs_to_chroma()