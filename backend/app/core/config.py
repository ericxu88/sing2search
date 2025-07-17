from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_index_name: str = "sing2search-index"

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True

    sample_rate: int = 22050
    max_audio_length: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()