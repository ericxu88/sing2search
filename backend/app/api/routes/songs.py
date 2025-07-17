from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging

from app.services.midi_service import MIDIService

logger = logging.getLogger(__name__)

router = APIRouter()

midi_service = MIDIService()

class SongUploadResponse(BaseModel):
    success: bool
    song_id: str
    message: str
    song_info: Optional[Dict] = None

class SearchResponse(BaseModel):
    success: bool
    query_info: Dict
    results: List[Dict]
    total_results: int

class StatsResponse(BaseModel):
    database: Dict
    embedder: Dict
    service: Dict

@router.post("/upload", response_model=SongUploadResponse)
async def upload_song(
    file: UploadFile = File(...),
    title: str = Form(...),
    artist: str = Form(...),
    album: Optional[str] = Form(None),
    genre: Optional[str] = Form(None),
    song_id: Optional[str] = Form(None)
):
    
    try:
     
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
       
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
    
        metadata = {
            "title": title.strip(),
            "artist": artist.strip(),
            "album": album.strip() if album else "",
            "genre": genre.strip() if genre else "",
            "filename": file.filename,
            "file_size": len(audio_bytes)
        }
        
        success, final_song_id, result_info = midi_service.add_song(
            audio_bytes=audio_bytes,
            metadata=metadata,
            song_id=song_id
        )
        
        if success:
            return SongUploadResponse(
                success=True,
                song_id=final_song_id,
                message=f"Successfully added '{title}' by {artist}",
                song_info=result_info
            )
        else:
            error_msg = result_info.get("error", "Unknown error")
            raise HTTPException(status_code=500, detail=f"Failed to add song: {error_msg}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_song: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/stats")
async def get_database_stats():
   
    try:
        stats = midi_service.get_database_stats()
        
        if "error" in stats:
            raise HTTPException(status_code=500, detail=f"Failed to get stats: {stats['error']}")
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/{song_id}")
async def get_song(song_id: str):

    try:
        song_info = midi_service.get_song_by_id(song_id)
        
        if song_info is None:
            raise HTTPException(status_code=404, detail=f"Song not found: {song_id}")
        
        return song_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting song {song_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/")
async def list_songs(limit: int = 10, offset: int = 0):

    try:
        stats = midi_service.get_database_stats()
        
        return {
            "total_songs": stats.get("database", {}).get("total_songs", 0),
            "limit": limit,
            "offset": offset,
            "message": "Use search endpoint to find specific songs"
        }
        
    except Exception as e:
        logger.error(f"Error listing songs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.delete("/{song_id}")
async def delete_song(song_id: str):

    try:
        raise HTTPException(
            status_code=501, 
            detail="Song deletion not implemented yet"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting song {song_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health/check")
async def health_check():
    try:
        is_healthy = midi_service.is_healthy()
        
        if is_healthy:
            return {
                "status": "healthy",
                "service": "songs",
                "message": "All systems operational"
            }
        else:
            raise HTTPException(
                status_code=503, 
                detail="Service unhealthy - check database connection and embedder"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")