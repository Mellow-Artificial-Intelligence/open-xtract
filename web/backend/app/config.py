from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application
    app_name: str = "OpenXtract API"
    debug: bool = False
    api_v1_prefix: str = "/api/v1"

    # Database
    database_url: str = "postgresql+asyncpg://openxtract:openxtract@localhost:5432/openxtract"
    database_url_sync: str = "postgresql://openxtract:openxtract@localhost:5432/openxtract"

    # Security
    secret_key: str = "change-me-in-production"
    encryption_key: str = "change-me-32-byte-key-for-aes256"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # MinIO
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "openxtract"
    minio_secret_key: str = "openxtract"
    minio_bucket: str = "openxtract-files"
    minio_secure: bool = False

    # Redis
    redis_url: str = "redis://localhost:6379"

    # CORS
    cors_origins: str = "http://localhost:3000"

    # OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    return Settings()
