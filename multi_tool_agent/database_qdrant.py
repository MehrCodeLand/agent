# database_qdrant.py
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

class QdrantDB:
    def __init__(
        self,
        url: str = "127.0.0.1",
        port: int = 6333,
        vector_size: int = 384,
        collection_name: str = "rag_collection",
    ):
        self.client = QdrantClient(url=url, port=port)
        self.collection_name = collection_name

        # (Re)create collection with cosine similarity
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=rest.VectorParams(
                size=vector_size, distance=rest.Distance.COSINE
            ),
        )

    def insert(self, ids: list[str], vectors: list[list[float]], payloads: list[dict]):
        """
        Bulk-insert vectors + payloads under given IDs.
        """
        points = [
            rest.PointStruct(id=i, vector=v, payload=p)
            for i, v, p in zip(ids, vectors, payloads)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: list[float], top_k: int = 5):
        """
        Returns top_k hits (each has .id, .score, .payload).
        """
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
