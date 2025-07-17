from pinecone import Pinecone
import numpy as np
from typing import Dict, Optional, List
import os
from app.core.config import settings

class PineconeClient:
    def __init__(self):
        self.api_key = settings.pinecone_api_key
        self.index_name = settings.pinecone_index_name
        self.pc = None
        self.index = None

        if self.api_key:
            self._initialize_client()
        
    def _initialize_client(self):
        try: 
            self.pc = Pinecone(api_key=self.api_key)
            self.index = self.pc.Index(self.index_name)
            print(f"Connected to Pinecone Index: {self.index_name}")

        except Exception as e:
            print(f"Failed to connect to Pinecone: {str(e)}")
            self.index = None

    def is_connected(self) -> bool:
        return self.index is not None
    
    def upsert_embedding(self, song_id: str, embedding: np.ndarray, metadata: Dict, namespace: str = "default") -> bool:
        #upload or update song in pinecone

        if not self.is_connected():
            return False
        
        try:
            self.index.upsert(vectors=[(song_id, embedding.tolist(), metadata)],
                              namespace=namespace)
            return True
        except Exception as e:
            print(f"Error upserting embedding: {str(e)}")
            return False
        
    def search_similar(self, embedding: np.ndarray, top_k: int = 5, namespace: str = "default", filter_dict: Optional[Dict] = None) -> List[Dict]:
        #searches db for similar audio embeddings
        if not self.is_connected():
            return []
        
        try:
            results = self.index.query(
                vector=embedding.tolist(),
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=filter_dict
            )

            return [{
                "id": match.id,
                "score": float(match.score),
                "metadata": match.metadata
            } for match in results.matches]
        
        except Exception as e:
            print(f"Error searching embeddings: {str(e)}")
            return []
        
    def get_stats(self, namespace: str = "default") -> Dict:
        if not self.is_connected():
            return {}
        
        try:
            stats = self.index.describe_index_stats()

            namespaces_dict = {}
            if stats.namespaces:
                for ns_name, ns_summary in stats.namespaces.items():
                    namespaces_dict[ns_name] = {
                        "vector_count": ns_summary.vector_count
                    }
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
                "namespaces": namespaces_dict
            }
        
        except Exception as e:
            print(f" Error getting stats for index: {str(e)}")
            return {"error": str(e)}
        