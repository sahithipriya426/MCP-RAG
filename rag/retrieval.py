from rag.reranker import DocumentReranker


reranker = DocumentReranker()


def retrieve_and_rerank(
    vectorstore,
    query: str,
    initial_k: int = 10,
    final_k: int = 3
) -> list:

    # 1. First-stage dense retrieval
    initial_docs = vectorstore.similarity_search(
        query,
        k=initial_k
    )

    if not initial_docs:
        return []

    # 2. Second-stage cross-encoder re-ranking
    reranked_docs = reranker.rerank(
        query,
        initial_docs,
        top_k=final_k
    )

    return reranked_docs


def build_context(documents: list) -> str:

    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')} "
        f"| Page: {doc.metadata.get('page', 'Unknown')}]\n"
        f"{doc.page_content.strip()}"
        for doc in documents
    )
