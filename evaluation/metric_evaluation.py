import json
import time
from pathlib import Path

import numpy as np

from rag.vectorstore import get_vectorstore
from rag.reranker import DocumentReranker
from evaluation.queries import BENCHMARK


DENSE_K = 20
RERANK_K = 10

RESULTS_FILE = Path(
    "evaluation/results/retrieval_results.json"
)


def source_name(source):
    return (
        str(source)
        .replace("\\", "/")
        .split("/")[-1]
        .lower()
    )


def page_key(doc):
    meta = doc.metadata or {}

    return (
        source_name(meta.get("source")),
        meta.get("page")
    )


def unique_pages(docs):
    """
    Collapse multiple chunks from the same
    source-page into one evaluation result.
    """

    unique = []
    seen = set()

    for doc in docs:
        key = page_key(doc)

        if key not in seen:
            seen.add(key)
            unique.append(doc)

    return unique


def relevant(doc, truth):
    meta = doc.metadata or {}

    return (
        source_name(meta.get("source"))
        == source_name(truth["source"])
        and meta.get("page") in truth["pages"]
    )


def mrr(docs, truth):
    docs = unique_pages(docs)

    for rank, doc in enumerate(docs, 1):

        if relevant(doc, truth):
            return 1.0 / rank

    return 0.0


def recall_at_k(docs, truth, k):
    docs = unique_pages(docs)

    return float(
        any(
            relevant(doc, truth)
            for doc in docs[:k]
        )
    )


def ndcg_at_k(docs, truth, k=10):
    docs = unique_pages(docs)[:k]

    hits = [
        int(relevant(doc, truth))
        for doc in docs
    ]

    dcg = sum(
        rel / np.log2(rank + 2)
        for rank, rel in enumerate(hits)
    )

    ideal = min(
        len(truth["pages"]),
        k
    )

    idcg = sum(
        1 / np.log2(rank + 2)
        for rank in range(ideal)
    )

    return dcg / idcg if idcg else 0.0


def evaluate(vs, reranker):

    results = []

    improved = 0
    degraded = 0
    unchanged = 0

    for i, truth in enumerate(BENCHMARK, 1):

        query = truth["query"]

        # -------------------------
        # Dense retrieval
        # -------------------------

        start = time.perf_counter()

        dense_docs = vs.similarity_search(
            query,
            k=DENSE_K
        )

        dense_latency = (
            time.perf_counter() - start
        ) * 1000

        # -------------------------
        # Cross-Encoder reranking
        # -------------------------

        start = time.perf_counter()

        reranked_docs = reranker.rerank(
            query,
            dense_docs,
            top_k=RERANK_K
        )

        rerank_latency = (
            time.perf_counter() - start
        ) * 1000

        # -------------------------
        # Metrics
        # -------------------------

        dense_mrr = mrr(
            dense_docs,
            truth
        )

        reranked_mrr = mrr(
            reranked_docs,
            truth
        )

        dense_ndcg = ndcg_at_k(
            dense_docs,
            truth
        )

        reranked_ndcg = ndcg_at_k(
            reranked_docs,
            truth
        )

        if reranked_mrr > dense_mrr:
            improved += 1

        elif reranked_mrr < dense_mrr:
            degraded += 1

        else:
            unchanged += 1

        result = {
            "query": query,
            "source": truth["source"],
            "ground_truth_pages": truth["pages"],

            "dense_mrr": dense_mrr,
            "reranked_mrr": reranked_mrr,

            "dense_recall@3": recall_at_k(
                dense_docs,
                truth,
                3
            ),

            "reranked_recall@3": recall_at_k(
                reranked_docs,
                truth,
                3
            ),

            "dense_recall@5": recall_at_k(
                dense_docs,
                truth,
                5
            ),

            "reranked_recall@5": recall_at_k(
                reranked_docs,
                truth,
                5
            ),

            "dense_recall@10": recall_at_k(
                dense_docs,
                truth,
                10
            ),

            "reranked_recall@10": recall_at_k(
                reranked_docs,
                truth,
                10
            ),

            "dense_ndcg@10": dense_ndcg,
            "reranked_ndcg@10": reranked_ndcg,

            "dense_latency_ms": dense_latency,
            "rerank_latency_ms": rerank_latency,
            "total_latency_ms": (
                dense_latency +
                rerank_latency
            )
        }

        results.append(result)

        print(
            f"{i:02d}. "
            f"MRR {dense_mrr:.3f} → "
            f"{reranked_mrr:.3f} | "
            f"NDCG {dense_ndcg:.3f} → "
            f"{reranked_ndcg:.3f}"
        )

    return (
        results,
        improved,
        degraded,
        unchanged
    )


def main():

    print("Loading RAG components...")

    vs = get_vectorstore("chroma_db")
    reranker = DocumentReranker()

    (
        results,
        improved,
        degraded,
        unchanged
    ) = evaluate(
        vs,
        reranker
    )

    metrics = [
        "dense_mrr",
        "reranked_mrr",

        "dense_recall@3",
        "reranked_recall@3",

        "dense_recall@5",
        "reranked_recall@5",

        "dense_recall@10",
        "reranked_recall@10",

        "dense_ndcg@10",
        "reranked_ndcg@10",

        "dense_latency_ms",
        "rerank_latency_ms",
        "total_latency_ms"
    ]

    summary = {
        "queries": len(results),

        **{
            metric: float(
                np.mean(
                    [r[metric] for r in results]
                )
            )
            for metric in metrics
        },

        "mrr_lift": 0.0,
        "ndcg_lift": 0.0,

        "reranking_improved": improved,
        "reranking_degraded": degraded,
        "reranking_unchanged": unchanged
    }

    summary["mrr_lift"] = (
        summary["reranked_mrr"]
        - summary["dense_mrr"]
    )

    summary["ndcg_lift"] = (
        summary["reranked_ndcg@10"]
        - summary["dense_ndcg@10"]
    )

    print("\n========== FINAL RESULTS ==========")

    for key, value in summary.items():

        if isinstance(value, float):
            print(
                f"{key}: {value:.4f}"
            )

        else:
            print(
                f"{key}: {value}"
            )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "evaluation_config": {
                    "dense_k": DENSE_K,
                    "rerank_k": RERANK_K,
                    "evaluation_unit":
                        "unique_source_pages"
                },
                "summary": summary,
                "queries": results
            },
            f,
            indent=2
        )

    print(
        f"\nSaved to {RESULTS_FILE}"
    )


if __name__ == "__main__":
    main()