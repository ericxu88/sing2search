from fastapi import APIRouter
from pydantic import BaseModel
import os


router = APIRouter()

class HealthResponse(BaseModel):
    status: str
    message: str

#health check endpoint
@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="Healthy",
        message="Sing2Search API is running",
    )

@router.get("/ping")
async def ping():
    return {"message": "pong"}