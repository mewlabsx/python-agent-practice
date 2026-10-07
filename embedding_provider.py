import os

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
_model = None


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(
            os.getenv("EMBEDDING_MODEL", MODEL_NAME), device="cpu",
            local_files_only=os.getenv("EMBEDDING_LOCAL_ONLY", "1") != "0",
        )
    return _model


def embed_text(text):
    if not isinstance(text, str):
        raise TypeError("Embedding input must be a string")
    return get_model().encode(text, normalize_embeddings=True).tolist()


def embed_texts(texts):
    if not isinstance(texts, list):
        raise TypeError("Embedding inputs must be a list")
    return [item.tolist() for item in get_model().encode(texts, normalize_embeddings=True)]
