import numpy as np
from numpy.linalg import norm

# Import your existing RAG components
from rag.vectorstore import get_vectorstore
from rag.reranker import DocumentReranker
from rag.embeddings import get_embeddings

def evaluate_reranker_mrr(
    queries: list[str], 
    ground_truth_ids: list[list[str]], 
    vectorstore, 
    reranker
) -> dict:
    """
    Computes MRR directly using NumPy matrix operations to mathematically 
    prove the performance lift of the Cross-Encoder over raw dense retrieval.
    """
    num_queries = len(queries)
    dense_mrr = np.zeros(num_queries)
    rerank_mrr = np.zeros(num_queries)
    
    for i, query in enumerate(queries):
        truth = set(ground_truth_ids[i])
        
        # 1. Evaluate Initial Dense Retrieval (k=10)
        initial_docs = vectorstore.similarity_search(query, k=10)
        initial_ids = [doc.metadata.get("source") for doc in initial_docs]
        
        dense_hits = np.isin(initial_ids, list(truth))
        if dense_hits.any():
            dense_mrr[i] = 1.0 / (np.argmax(dense_hits) + 1)
            
        # 2. Evaluate Cross-Encoder Reranking (k=3)
        reranked_docs = reranker.rerank(query, initial_docs, top_k=3)
        reranked_ids = [doc.metadata.get("source") for doc in reranked_docs]
        
        rerank_hits = np.isin(reranked_ids, list(truth))
        if rerank_hits.any():
            rerank_mrr[i] = 1.0 / (np.argmax(rerank_hits) + 1)
            
    return {
        "dense_mrr": np.mean(dense_mrr),
        "reranked_mrr": np.mean(rerank_mrr),
        "lift": np.mean(rerank_mrr) - np.mean(dense_mrr)
    }

def evaluate_context_relevance(query: str, retrieved_docs: list) -> float:
    """
    Evaluates contextual relevance via Cosine Similarity between the query 
    and the retrieved context chunk: (A · B) / (||A|| * ||B||)
    """
    if not retrieved_docs:
        return 0.0
        
    embeddings_model = get_embeddings()
    
    # Generate 1D array for the query
    query_vector = np.array(embeddings_model.embed_query(query))
    
    relevance_scores = np.zeros(len(retrieved_docs))
    
    for i, doc in enumerate(retrieved_docs):
        # Generate 1D array for the document chunk
        doc_vector = np.array(embeddings_model.embed_query(doc.page_content))
        
        # Compute Cosine Similarity strictly from scratch
        dot_product = np.dot(query_vector, doc_vector)
        norm_q = norm(query_vector)
        norm_d = norm(doc_vector)
        
        if norm_q > 0 and norm_d > 0:
            relevance_scores[i] = dot_product / (norm_q * norm_d)
            
    # Return the mean cosine similarity of all retrieved chunks
    return np.mean(relevance_scores)


if __name__ == "__main__":
    # 1. Initialize your existing components
    # Assuming this script is in evals/, we point one directory up to chroma_db
    print("Initializing vectorstore and models...")
    vs = get_vectorstore("chroma_db") 
    reranker = DocumentReranker()
    
    # 2. Define your test dataset
    # Replace these with actual queries relevant to your PDFs
    test_queries = [
        "What is the model's architecture?",
        "Explain the training methodology."
    ]
    
    # Replace these with the actual deterministic source filenames your ingestion created
    ground_truth = [
        ["Improved PINN for Burgers' Equations.pdf"], 
        ["Lecture notes on combinatorics.pdf"],
        ["mcp-architecture-notes.pdf"],
        ["MLBOOK.pdf"]
    ]
    
    # 3. Run MRR Evaluation
    print("\n--- Running Retrieval Pipeline Evaluation ---")
    mrr_results = evaluate_reranker_mrr(test_queries, ground_truth, vs, reranker)
    print(f"Initial Dense MRR:  {mrr_results['dense_mrr']:.4f}")
    print(f"Reranked MRR:       {mrr_results['reranked_mrr']:.4f}")
    print(f"Performance Lift:   {mrr_results['lift']:+.4f}")
    
    # 4. Run Context Relevance Evaluation
    print("\n--- Running Context Relevance Evaluation ---")
    query = test_queries[0]
    
    # Manually trigger your 2-stage pipeline to get the final docs
    initial = vs.similarity_search(query, k=10)
    final_docs = reranker.rerank(query, initial, top_k=3)
    
    # Measure the mathematical similarity
    relevance = evaluate_context_relevance(query, final_docs)
    print(f"Mean Cosine Similarity for query 1: {relevance:.4f}")