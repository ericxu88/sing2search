import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import uuid
from datetime import datetime
import os

from app.models.midi_embedder import MIDIEmbedder
from app.core.database import PineconeClient
from app.core.config import settings

logger = logging.getLogger(__name__)

class MIDIService:
    
    def __init__(self):
        self.embedder = MIDIEmbedder()
        self.db_client = PineconeClient()
        self.namespace = "songs" 
        self.min_similarity_threshold = 0.6 
        
        logger.info("MIDI Service initialized")

    def add_song(self, 
                 audio_bytes: bytes, 
                 metadata: Dict, 
                 song_id: Optional[str] = None) -> Tuple[bool, str, Dict]:

        try:
            if song_id is None:
                song_id = str(uuid.uuid4())
            
            logger.info(f"Adding song: {metadata.get('title', 'Unknown')} by {metadata.get('artist', 'Unknown')}")
            
            embedding = self.embedder.generate_embedding_from_bytes(audio_bytes)
            
            if embedding is None:
                return False, song_id, {"error": "Failed to generate embedding from audio"}
       
            enhanced_metadata = {
                **metadata,
                "added_at": datetime.now().isoformat(),
                "embedding_model": "MIDI_Embedder",
                "embedding_dimensions": 128,
                "processing_version": "1.0"
            }
            
            
            success = self.db_client.upsert_embedding(
                song_id=song_id,
                embedding=embedding,
                metadata=enhanced_metadata,
                namespace=self.namespace
            )
            
            if success:
                result_info = {
                    "song_id": song_id,
                    "title": metadata.get("title"),
                    "artist": metadata.get("artist"),
                    "embedding_stats": {
                        "dimensions": len(embedding),
                        "non_zero_features": int(np.count_nonzero(embedding)),
                        "l2_norm": float(np.linalg.norm(embedding))
                    }
                }
                logger.info(f"Successfully added song: {song_id}")
                return True, song_id, result_info
            else:
                return False, song_id, {"error": "Failed to store in database"}
                
        except Exception as e:
            logger.error(f"Error adding song: {str(e)}")
            return False, song_id, {"error": str(e)}

    def search_by_audio(self, 
                       audio_bytes: bytes, 
                       top_k: int = 5,
                       min_similarity: Optional[float] = None) -> Tuple[bool, List[Dict]]:
    
        try:
            if min_similarity is None:
                min_similarity = self.min_similarity_threshold
            
            logger.info("Searching for similar songs from audio input")
            
            query_embedding = self.embedder.generate_embedding_from_bytes(audio_bytes)
            
            if query_embedding is None:
                return False, [{"error": "Failed to generate embedding from audio"}]
            

            raw_results = self.db_client.search_similar(
                embedding=query_embedding,
                top_k=top_k * 2, 
                namespace=self.namespace
            )
        
            filtered_results = []
            for result in raw_results:
                if result['score'] >= min_similarity:
                    formatted_result = {
                        "song_id": result['id'],
                        "similarity_score": result['score'],
                        "title": result['metadata'].get('title', 'Unknown'),
                        "artist": result['metadata'].get('artist', 'Unknown'),
                        "album": result['metadata'].get('album', ''),
                        "genre": result['metadata'].get('genre', ''),
                        "added_at": result['metadata'].get('added_at', ''),
                        "match_quality": self._get_match_quality(result['score'])
                    }
                    filtered_results.append(formatted_result)
            
            filtered_results = filtered_results[:top_k]
            
            logger.info(f"Found {len(filtered_results)} similar songs above threshold {min_similarity}")
            return True, filtered_results
            
        except Exception as e:
            logger.error(f"Error searching songs: {str(e)}")
            return False, [{"error": str(e)}]

    def get_song_by_id(self, song_id: str) -> Optional[Dict]:
    
        try:
            if not self.db_client.is_connected():
                return None
                
            fetch_response = self.db_client.index.fetch(
                ids=[song_id], 
                namespace=self.namespace
            )
            
            if song_id in fetch_response.vectors:
                vector_data = fetch_response.vectors[song_id]
                metadata = vector_data.metadata
                
                return {
                    "song_id": song_id,
                    "title": metadata.get('title'),
                    "artist": metadata.get('artist'),
                    "album": metadata.get('album'),
                    "genre": metadata.get('genre'),
                    "added_at": metadata.get('added_at'),
                    "metadata": metadata
                }
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error getting song by ID {song_id}: {str(e)}")
            return None

    def get_database_stats(self) -> Dict:
    
        try:
            db_stats = self.db_client.get_stats()
            embedder_info = self.embedder.get_embedding_info()
            
            return {
                "database": {
                    "total_songs": db_stats.get("total_vectors", 0),
                    "dimensions": db_stats.get("dimension", 128),
                    "namespaces": db_stats.get("namespaces", {}),
                    "main_namespace_songs": db_stats.get("namespaces", {}).get(self.namespace, {}).get("vector_count", 0)
                },
                "embedder": embedder_info,
                "service": {
                    "namespace": self.namespace,
                    "min_similarity_threshold": self.min_similarity_threshold,
                    "status": "active"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {str(e)}")
            return {"error": str(e)}

    def _get_match_quality(self, score: float) -> str:
    
        if score >= 0.9:
            return "Excellent"
        elif score >= 0.8:
            return "Very Good"
        elif score >= 0.7:
            return "Good"
        elif score >= 0.6:
            return "Fair"
        else:
            return "Poor"

    def is_healthy(self) -> bool:
    
        try:
            
            if not self.db_client.is_connected():
                return False
            
            embedder_info = self.embedder.get_embedding_info()
            if not embedder_info.get('basic_pitch_available', False):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    def batch_add_songs(self, songs_data: List[Dict]) -> Dict:
    
        results = {
            "successful": [],
            "failed": [],
            "total": len(songs_data)
        }
        
        for i, song_data in enumerate(songs_data):
            try:
                audio_bytes = song_data['audio_bytes']
                metadata = song_data['metadata']
                song_id = song_data.get('song_id')
                
                success, final_song_id, result_info = self.add_song(
                    audio_bytes=audio_bytes,
                    metadata=metadata,
                    song_id=song_id
                )
                
                if success:
                    results["successful"].append({
                        "index": i,
                        "song_id": final_song_id,
                        "title": metadata.get("title"),
                        "result_info": result_info
                    })
                else:
                    results["failed"].append({
                        "index": i,
                        "song_id": final_song_id,
                        "title": metadata.get("title"),
                        "error": result_info.get("error")
                    })
                    
            except Exception as e:
                results["failed"].append({
                    "index": i,
                    "error": str(e)
                })
        
        logger.info(f"Batch processing complete: {len(results['successful'])}/{results['total']} successful")
        return results