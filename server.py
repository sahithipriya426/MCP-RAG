import os
from glob import glob

from fastmcp import FastMCP

from rag.generation import generate_answer
from rag.retrieval import build_context, retrieve_and_rerank
from rag.vectorstore import get_vectorstore, ingest_pdfs_to_chroma

mcp = FastMCP("Production-RAG-Server")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

_VECTORSTORE = None


def get_or_init_vectorstore(force_reload: bool = False):
    """Lazy loader for persistent Chroma vectorstore singleton."""
    global _VECTORSTORE
    if _VECTORSTORE is None or force_reload:
        _VECTORSTORE = get_vectorstore(CHROMA_DIR)
    return _VECTORSTORE


@mcp.tool()
def search_documents(query: str, top_k: int = 3) -> str:
    """
    Search indexed PDF context using BGE retrieval and Cross-Encoder reranking.
    """
    vs = get_or_init_vectorstore()
    if vs is None:
        return "Knowledge base is empty. Please add and index documents."

    ranked_docs = retrieve_and_rerank(vs, query, initial_k=10, final_k=top_k)
    if not ranked_docs:
        return "No sufficiently relevant documents found matching the query."

    return build_context(ranked_docs)


@mcp.tool()
def ask_document_question(query: str, top_k: int = 3) -> str:
    """
    Retrieve relevant PDF passages and generate a grounded answer with source attribution.
    """
    vs = get_or_init_vectorstore()
    if vs is None:
        return "Knowledge base is empty. Please index documents first."

    ranked_docs = retrieve_and_rerank(vs, query, initial_k=10, final_k=top_k)
    if not ranked_docs:
        return "The answer could not be found in the provided documents."

    return generate_answer(query, ranked_docs)


@mcp.tool()
def index_documents() -> str:
    """
    Scan the data directory and update ChromaDB idempotently.
    """
    pdf_paths = glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_paths:
        return "No PDFs found in the data directory."

    status = ingest_pdfs_to_chroma(CHROMA_DIR, pdf_paths)
    get_or_init_vectorstore(force_reload=True)
    return f"Index updated: {status}"


@mcp.tool()
def list_available_documents() -> list[str]:
    """
    List PDF files available in the data directory.
    """
    pdf_paths = glob(os.path.join(DATA_DIR, "*.pdf"))
    return [os.path.basename(path) for path in pdf_paths]


if __name__ == "__main__":
    # Utilizing secure STDIO transport for zero-network overhead 
    # local communication with Claude Desktop / Cursor / Inspector
    mcp.run(transport="stdio")