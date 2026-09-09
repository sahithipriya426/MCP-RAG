import json
import hashlib
import os
import re
from pathlib import Path

from rag.vectorstore import get_vectorstore
from evaluation.queries import BENCHMARK


DENSE_K = 20
MIN_OVERLAP = 2

OUTPUT_FILE = Path(
    "evaluation/results/ground_truth.json"
)


STOPWORDS = {
    "what", "are", "is", "the", "a", "an", "of", "to",
    "in", "for", "how", "why", "does", "do", "can",
    "and", "or", "on", "with", "from", "used", "use",
    "their", "they", "this", "that", "be", "as"
}


def make_doc_key(doc):
    source = os.path.basename(
        doc.metadata.get("source", "unknown")
    )

    text_hash = hashlib.md5(
        doc.page_content.strip().encode("utf-8")
    ).hexdigest()[:12]

    return f"{source}_{text_hash}"


def tokenize(text):
    words = re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )

    return {
        word
        for word in words
        if word not in STOPWORDS
        and len(word) > 2
    }


def relevance_score(query, text):
    query_words = tokenize(query)
    text_words = tokenize(text)

    if not query_words:
        return 0.0

    overlap = query_words & text_words

    return len(overlap) / len(query_words)


def main():

    print("Loading vectorstore...")

    vs = get_vectorstore("chroma_db")

    count = vs._collection.count()

    print(f"Chroma documents: {count}")

    if count == 0:
        print("ERROR: ChromaDB is empty.")
        print("Run ingestion first.")
        return

    results = []

    for i, item in enumerate(BENCHMARK, 1):

        query = item["query"]
        source = os.path.basename(item["source"])

        docs = vs.similarity_search(
            query,
            k=DENSE_K
        )

        candidates = []

        for rank, doc in enumerate(docs, 1):

            doc_source = os.path.basename(
                doc.metadata.get(
                    "source",
                    "unknown"
                )
            )

            overlap = relevance_score(
                query,
                doc.page_content
            )

            # Only chunks from the expected document
            # can be relevant.
            relevant = (
                doc_source == source
                and overlap >= MIN_OVERLAP / max(
                    len(tokenize(query)),
                    1
                )
            )

            candidates.append({
                "rank": rank,
                "doc_key": make_doc_key(doc),
                "source": doc_source,
                "text": doc.page_content,
                "relevance_score": overlap,
                "relevant": relevant
            })

        relevant_keys = [
            c["doc_key"]
            for c in candidates
            if c["relevant"]
        ]

        results.append({
            "query": query,
            "source": source,
            "candidates": candidates,
            "relevant_doc_keys": relevant_keys
        })

        print(
            f"{i:02d}/50 | "
            f"Candidates: {len(candidates)} | "
            f"Relevant: {len(relevant_keys)}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()