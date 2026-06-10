from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)


class VectorDBAdapter(ABC):
    @abstractmethod
    def add_embeddings(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict]] = None) -> str:
        pass

    @abstractmethod
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        pass

    @abstractmethod
    def delete_collection(self) -> None:
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        pass


class ChromaAdapter(VectorDBAdapter):
    def __init__(self, persist_directory: str = "./chroma_db"):
        import chromadb
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = None

    def create_collection(self, name: str = "rag_eval", embedding_dim: int = 1536):
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )
        return name

    def add_embeddings(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict]] = None) -> str:
        if self.collection is None:
            self.create_collection()
        
        ids = [f"doc_{i}" for i in range(len(texts))]
        metadatas = metadata if metadata else None
        
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        return f"Added {len(texts)} documents"

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        if self.collection is None:
            return []
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return [
            (doc, float(dist), meta)
            for doc, dist, meta in zip(
                results["documents"][0],
                results["distances"][0],
                results["metadatas"][0] if results["metadatas"] else [{}]
            )
        ]

    def delete_collection(self) -> None:
        if self.collection:
            self.client.delete_collection(self.collection.name)
            self.collection = None

    def get_collection_stats(self) -> Dict[str, Any]:
        if self.collection is None:
            return {"count": 0}
        return {"count": self.collection.count()}


class QdrantAdapter(VectorDBAdapter):
    def __init__(self, url: str = "http://localhost:6333", api_key: str = "", collection_name: str = "rag_eval"):
        from qdrant_client import QdrantClient
        self.url = url
        self.api_key = api_key
        self.collection_name = collection_name
        self.client = QdrantClient(url=url, api_key=api_key if api_key else None)
        self._embedding_dim = 1536

    def create_collection(self, embedding_dim: int = 1536):
        from qdrant_client.models import Distance, VectorParams
        self._embedding_dim = embedding_dim
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=embedding_dim, distance=Distance.COSINE)
        )

    def add_embeddings(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict]] = None) -> str:
        from qdrant_client.models import PointStruct, Payload
        
        if not texts:
            return "No documents to add"
        
        if not self.client.collection_exists(self.collection_name):
            self.create_collection(embedding_dim=len(embeddings[0]))
        
        points = [
            PointStruct(
                id=i,
                vector=emb,
                payload={"text": text, "metadata": meta or {}}
            )
            for i, (text, emb, meta) in enumerate(zip(texts, embeddings, metadata or [{} for _ in texts]))
        ]
        
        self.client.upsert(collection_name=self.collection_name, points=points)
        return f"Added {len(texts)} documents"

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        from qdrant_client.models import Filter, SearchParams
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            search_params=SearchParams(hnsw_ef=128)
        )
        
        return [
            (hit.payload.get("text", ""), float(hit.score), hit.payload.get("metadata", {}))
            for hit in results
        ]

    def delete_collection(self) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.client.collection_exists(self.collection_name):
            return {"count": 0}
        info = self.client.get_collection(self.collection_name)
        return {"count": info.points_count}


class FAISSAdapter(VectorDBAdapter):
    def __init__(self, embedding_dim: int = 1536):
        import faiss
        self.embedding_dim = embedding_dim
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.texts = []
        self.metadata = []

    def add_embeddings(self, texts: List[str], embeddings: List[List[float]], metadata: Optional[List[Dict]] = None) -> str:
        if not texts:
            return "No documents to add"
        
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        
        self.index.add(embeddings_array)
        self.texts.extend(texts)
        self.metadata.extend(metadata if metadata else [{} for _ in texts])
        
        return f"Added {len(texts)} documents"

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        if self.index.ntotal == 0:
            return []
        
        query = np.array([query_embedding]).astype('float32')
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, min(top_k, self.index.ntotal))
        
        return [
            (self.texts[idx], float(dist), self.metadata[idx])
            for dist, idx in zip(distances[0], indices[0])
            if idx < len(self.texts)
        ]

    def delete_collection(self) -> None:
        import faiss
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.texts = []
        self.metadata = []

    def get_collection_stats(self) -> Dict[str, Any]:
        return {"count": self.index.ntotal}


def get_vector_db_adapter(provider: str, **kwargs) -> VectorDBAdapter:
    from app.config import get_settings
    settings = get_settings()

    adapters = {
        "chroma": ChromaAdapter(persist_directory=kwargs.get("persist_directory", "./chroma_db")),
        "qdrant": QdrantAdapter(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            collection_name=settings.qdrant_collection_name
        ),
        "faiss": FAISSAdapter(embedding_dim=kwargs.get("embedding_dim", 1536)),
    }
    
    return adapters.get(provider.lower(), adapters["chroma"])
