import re
import sys
from rag.reranker import DocumentReranker

# Initialize as None to protect the startup handshake
_reranker = None

def get_reranker():
    """Lazy loader for the DocumentReranker."""
    global _reranker
    if _reranker is None:
        _reranker = DocumentReranker()
    return _reranker

def is_math_or_symbolic(query: str) -> bool:
    """
    Acts as a fast classification gate. Returns True if the query targets 
    discrete math, preventing tokenizer fragmentation in the cross-encoder.
    """
    math_keywords = {
        "formula", "combinatorics", "identity", "equation", 
        "theorem", "pascal", "subset", "multiplication", "addition"
    }
    query_lower = query.lower()
    if any(kw in query_lower for kw in math_keywords):
        return True
        
    symbolic_pattern = re.compile(r'[\[\]\{\}\(\)\+\-\*\/\=\^]')
    
    if len(symbolic_pattern.findall(query)) >= 2:
        return True
        
    return False

def retrieve_and_rerank(vectorstore, query: str, initial_k: int = 20, final_k: int = 3) -> list:
    """
    Executes dynamic dual-stage retrieval. Routes symbolic queries 
    away from the reranker to preserve precision and save CPU cycles.
    """
    initial_docs = vectorstore.similarity_search(query, k=initial_k)

    if not initial_docs:
        return []

    if is_math_or_symbolic(query):
        # Print to stderr to prevent corrupting the JSON-RPC STDIO stream
        print(f"\n⚡ Dynamic Route: Symbolic/Math query detected. Bypassing reranker.", file=sys.stderr)
        return initial_docs[:final_k]

    # Fetch the reranker dynamically only when needed
    reranked_docs = get_reranker().rerank(query, initial_docs, top_k=final_k)

    return reranked_docs

def build_context(documents: list) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'Unknown')} | Page: {doc.metadata.get('page', 'Unknown')}]\n{doc.page_content.strip()}"
        for doc in documents
    )