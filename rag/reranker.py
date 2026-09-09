from sentence_transformers import CrossEncoder


class DocumentReranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base"
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

        pairs = [
            [query, doc.page_content]
            for doc in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=16
        )

        scored_docs = list(zip(documents, scores))

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc
            for doc, _ in scored_docs[:top_k]
        ]