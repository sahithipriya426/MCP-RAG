import json
import time
import hashlib
import os
from pathlib import Path
import numpy as np

from rag.vectorstore import get_vectorstore
from rag.reranker import DocumentReranker

DENSE_K = 20
RERANK_K = 10

GROUND_TRUTH_FILE = Path("evaluation/results/ground_truth_labeled.json")
RESULTS_FILE = Path("evaluation/results/final_retrieval_results.json")


def mrr(ranks: list[int]) -> float:
    return 1.0 / min(ranks) if ranks else 0.0


def recall_at_k(ranks: list[int], k: int) -> float:
    return float(any(rank <= k for rank in ranks))


def dcg(ranks: list[int], k: int) -> float:
    return sum(1.0 / np.log2(rank + 1) for rank in ranks if rank <= k)


def ndcg_at_k(ranks: list[int], total_relevant: int, k: int = 10) -> float:
    if total_relevant == 0:
        return 0.0
    actual = dcg(ranks, k)
    ideal = sum(1.0 / np.log2(rank + 1) for rank in range(1, min(total_relevant, k) + 1))
    return actual / ideal if ideal else 0.0


def make_doc_key(doc) -> str:
    """Matches the 12-char deterministic MD5 hash key from build_ground_truth."""
    source = os.path.basename(doc.metadata.get("source", "unknown"))
    text_hash = hashlib.md5(doc.page_content.strip().encode("utf-8")).hexdigest()[:12]
    return f"{source}_{text_hash}"


def main():
    if not GROUND_TRUTH_FILE.exists():
        print(f"Error: {GROUND_TRUTH_FILE} not found. Run labeling steps first.")
        return

    print("Loading Ground Truth and Initializing Models...")
    with open(GROUND_TRUTH_FILE, "r", encoding="utf-8") as f:
        ground_truth = json.load(f)

    vs = get_vectorstore("chroma_db")
    reranker = DocumentReranker()

    results = []
    improved = 0
    degraded = 0
    unchanged = 0

    for i, item in enumerate(ground_truth, 1):
        query = item["query"]
        relevant_keys = set(item.get("relevant_doc_keys", []))

        if not relevant_keys:
            print(f"{i:02d}. Skipping: No relevant candidates labeled.")
            continue

        # 1. Live Dense Retrieval
        start_dense = time.perf_counter()
        dense_docs = vs.similarity_search(query, k=DENSE_K)
        dense_latency_ms = (time.perf_counter() - start_dense) * 1000

        dense_relevance = [
            rank for rank, doc in enumerate(dense_docs, 1)
            if make_doc_key(doc) in relevant_keys
        ]

        # 2. Live Cross-Encoder Reranking
        start_rerank = time.perf_counter()
        reranked_docs = reranker.rerank(query, dense_docs, top_k=RERANK_K)
        rerank_latency_ms = (time.perf_counter() - start_rerank) * 1000

        reranked_relevance = [
            rank for rank, doc in enumerate(reranked_docs, 1)
            if make_doc_key(doc) in relevant_keys
        ]

        # 3. Compute Metrics
        dense_mrr_val = mrr(dense_relevance)
        rerank_mrr_val = mrr(reranked_relevance)
        total_rel = len(relevant_keys)

        dense_ndcg_val = ndcg_at_k(dense_relevance, total_rel, k=10)
        rerank_ndcg_val = ndcg_at_k(reranked_relevance, total_rel, k=10)

        if rerank_mrr_val > dense_mrr_val:
            improved += 1
        elif rerank_mrr_val < dense_mrr_val:
            degraded += 1
        else:
            unchanged += 1

        results.append({
            "query_id": i,
            "query": query,
            "source": item["source"],
            "dense_mrr": dense_mrr_val,
            "reranked_mrr": rerank_mrr_val,
            "dense_recall@3": recall_at_k(dense_relevance, 3),
            "reranked_recall@3": recall_at_k(reranked_relevance, 3),
            "dense_recall@5": recall_at_k(dense_relevance, 5),
            "reranked_recall@5": recall_at_k(reranked_relevance, 5),
            "dense_ndcg@10": dense_ndcg_val,
            "reranked_ndcg@10": rerank_ndcg_val,
            "dense_latency_ms": dense_latency_ms,
            "rerank_latency_ms": rerank_latency_ms
        })

        print(
            f"{i:02d}. MRR {dense_mrr_val:.3f} -> {rerank_mrr_val:.3f} | "
            f"NDCG@10 {dense_ndcg_val:.3f} -> {rerank_ndcg_val:.3f}"
        )

    if not results:
        print("\nNo evaluable queries processed.")
        return

    # 4. Summary Aggregation
    summary = {
        "queries_evaluated": len(results),
        "dense_mrr": float(np.mean([r["dense_mrr"] for r in results])),
        "reranked_mrr": float(np.mean([r["reranked_mrr"] for r in results])),
        "mrr_lift": 0.0,
        "dense_recall@3": float(np.mean([r["dense_recall@3"] for r in results])),
        "reranked_recall@3": float(np.mean([r["reranked_recall@3"] for r in results])),
        "dense_recall@5": float(np.mean([r["dense_recall@5"] for r in results])),
        "reranked_recall@5": float(np.mean([r["reranked_recall@5"] for r in results])),
        "dense_ndcg@10": float(np.mean([r["dense_ndcg@10"] for r in results])),
        "reranked_ndcg@10": float(np.mean([r["reranked_ndcg@10"] for r in results])),
        "ndcg_lift": 0.0,
        "avg_dense_latency_ms": float(np.mean([r["dense_latency_ms"] for r in results])),
        "avg_rerank_latency_ms": float(np.mean([r["rerank_latency_ms"] for r in results])),
        "reranking_improved": improved,
        "reranking_degraded": degraded,
        "reranking_unchanged": unchanged
    }

    summary["mrr_lift"] = summary["reranked_mrr"] - summary["dense_mrr"]
    summary["ndcg_lift"] = summary["reranked_ndcg@10"] - summary["dense_ndcg@10"]

    print("\n========== FINAL RESULTS ==========")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")

    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "queries": results}, f, indent=2)

    print(f"\nSaved results to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()