"""MinIO storage service for file uploads."""

import io
import uuid
from datetime import timedelta

from minio import Minio
from minio.error import S3Error

from app.config import get_settings

settings = get_settings()


class StorageService:
    def __init__(self):
        self.client = Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            secure=settings.minio_secure,
        )
        self.bucket = settings.minio_bucket
        self._ensure_bucket()

    def _ensure_bucket(self) -> None:
        """Create bucket if it doesn't exist."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
        except S3Error as e:
            if e.code != "BucketAlreadyOwnedByYou":
                raise

    def upload_file(
        self,
        file_data: bytes,
        file_name: str,
        content_type: str,
    ) -> str:
        """Upload a file to MinIO. Returns the object path."""
        file_id = str(uuid.uuid4())
        object_name = f"{file_id}/{file_name}"

        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(file_data),
            length=len(file_data),
            content_type=content_type,
        )

        return object_name

    def get_file(self, object_path: str) -> bytes:
        """Download a file from MinIO."""
        response = self.client.get_object(self.bucket, object_path)
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def get_presigned_url(self, object_path: str, expires: timedelta = timedelta(hours=1)) -> str:
        """Get a presigned URL for file download."""
        return self.client.presigned_get_object(
            self.bucket,
            object_path,
            expires=expires,
        )

    def delete_file(self, object_path: str) -> None:
        """Delete a file from MinIO."""
        self.client.remove_object(self.bucket, object_path)

    def file_exists(self, object_path: str) -> bool:
        """Check if a file exists."""
        try:
            self.client.stat_object(self.bucket, object_path)
            return True
        except S3Error:
            return False


_storage_service: StorageService | None = None


def get_storage_service() -> StorageService:
    """Get or create storage service instance."""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
