from sentence_transformers import CrossEncoder

class DocumentReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        self.model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        documents: list,
        top_k: int = 3
    ) -> list:

        if not documents:
            return []

        # Construct query-document pairs
        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        # Compute relevance scores
        scores = self.model.predict(
            pairs,
            batch_size=16
        )

        # Rank documents by relevance
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, _ in scored_docs[:top_k]
        ]
