# embedding.py
from sentence_transformers import SentenceTransformer

class Embedder:
    """
    Free, on-device embedder using Sentence-Transformers.
    Default: all-MiniLM-L6-v2 (384-dim).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """
        Returns list of embedding vectors.
        """
        return self.model.encode(texts, convert_to_numpy=True).tolist()
