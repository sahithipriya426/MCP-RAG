from langchain_huggingface import HuggingFaceEmbeddings


def get_embeddings():
    """
    BAAI/bge-small-en-v1.5: 33.4M params, 384 dims.
    Fast CPU inference with contrastive normalization.
    """
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 64
        }
    )