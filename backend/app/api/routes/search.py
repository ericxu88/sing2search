from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging

from app.services.midi_service import MIDIService

logger = logging.getLogger(__name__)

router = APIRouter()

midi_service = MIDIService()

class SearchResponse(BaseModel):
    success: bool
    query_info: Dict
    results: List[Dict]
    total_results: int
    search_time_ms: Optional[float] = None

class SearchResult(BaseModel):
    song_id: str
    similarity_score: float
    title: str
    artist: str
    album: Optional[str] = ""
    genre: Optional[str] = ""
    match_quality: str

@router.post("/sing", response_model=SearchResponse)
async def search_by_singing(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    min_similarity: float = Query(0.6, ge=0.0, le=1.0, description="Minimum similarity threshold")
):

    try:
        import time
        start_time = time.time()
        
        if not file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
    
        allowed_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg'}
        file_extension = '.' + file.filename.split('.')[-1].lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file_extension}. Supported: {allowed_extensions}"
            )
    
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
    
        logger.info(f"Search query: {file.filename}, size: {len(audio_bytes)} bytes, top_k: {top_k}")
        
        success, results = midi_service.search_by_audio(
            audio_bytes=audio_bytes,
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        search_time = (time.time() - start_time) * 1000
        
        if success:
            query_info = {
                "filename": file.filename,
                "file_size_bytes": len(audio_bytes),
                "file_type": file_extension,
                "parameters": {
                    "top_k": top_k,
                    "min_similarity": min_similarity
                }
            }
            
            return SearchResponse(
                success=True,
                query_info=query_info,
                results=results,
                total_results=len(results),
                search_time_ms=search_time
            )
        else:
            error_msg = results[0].get("error", "Unknown search error") if results else "Search failed"
            raise HTTPException(status_code=500, detail=f"Search failed: {error_msg}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in search_by_humming: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/similar")
async def search_similar_to_song(
    song_id: str,
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    min_similarity: float = Query(0.6, ge=0.0, le=1.0, description="Minimum similarity threshold")
):
    try:
        raise HTTPException(
            status_code=501,
            detail="Similar song search not implemented yet"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search_similar_to_song: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/test")
async def test_search_system():

    try:

        import numpy as np
        import soundfile as sf
        import tempfile
        import os
        
        sr = 22050
        freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
        
        audio = np.array([])
        for freq in freqs:
            t = np.linspace(0, 0.5, int(sr * 0.5))
            note = 0.3 * np.sin(2 * np.pi * freq * t)
            audio = np.concatenate([audio, note])
    
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp_file:
            sf.write(tmp_file.name, audio, sr)
            tmp_file.seek(0)
            audio_bytes = tmp_file.read()
        
 
        success, results = midi_service.search_by_audio(
            audio_bytes=audio_bytes,
            top_k=3,
            min_similarity=0.3  # Lower threshold for testing
        )
        
        return {
            "test_audio": "C major scale",
            "search_successful": success,
            "results_found": len(results) if success else 0,
            "results": results if success else [],
            "message": "Test search completed"
        }
        
    except Exception as e:
        logger.error(f"Error in test search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Test search failed: {str(e)}")

@router.get("/health/check")
async def search_health_check():
    try:
        is_healthy = midi_service.is_healthy()
        
        if is_healthy:
            return {
                "status": "healthy",
                "service": "search",
                "message": "Search system operational",
                "components": {
                    "embedder": "available",
                    "database": "connected"
                }
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Search service unhealthy - check embedder and database"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@router.get("/stats")
async def get_search_stats():
    try:
        stats = midi_service.get_database_stats()
        
        search_stats = {
            **stats,
            "search_info": {
                "default_min_similarity": 0.6,
                "max_results_per_query": 20,
                "supported_audio_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg"],
                "average_search_time_estimate": "100-500ms"
            }
        }
        
        return search_stats
        
    except Exception as e:
        logger.error(f"Error getting search stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")