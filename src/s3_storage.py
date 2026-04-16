import io
import json

import structlog
from minio import Minio
from minio.error import S3Error

from src.config import settings

logger = structlog.get_logger()


class S3Storage:
    def __init__(self):
        self.client = Minio(
            settings.s3_endpoint,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            secure=settings.s3_use_ssl,
        )
        self.bucket = settings.s3_bucket
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self.client.bucket_exists(self.bucket):
            self.client.make_bucket(self.bucket)
            logger.info("created_bucket", bucket=self.bucket)

    def put_json_object(self, key: str, data: dict) -> None:
        body = json.dumps(data, default=str).encode("utf-8")
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=key,
            data=io.BytesIO(body),
            length=len(body),
            content_type="application/json",
        )
        logger.debug("put_json_object", key=key, size=len(body))

    def get_json_object(self, key: str) -> dict:
        response = self.client.get_object(
            bucket_name=self.bucket,
            object_name=key,
        )
        try:
            return json.loads(response.read())
        finally:
            response.close()
            response.release_conn()

    def put_bytes_object(self, key: str, data: bytes) -> None:
        self.client.put_object(
            bucket_name=self.bucket,
            object_name=key,
            data=io.BytesIO(data),
            length=len(data),
            content_type="application/octet-stream",
        )
        logger.debug("put_bytes_object", key=key, size=len(data))

    def get_bytes_object(self, key: str) -> bytes:
        response = self.client.get_object(
            bucket_name=self.bucket,
            object_name=key,
        )
        try:
            return response.read()
        finally:
            response.close()
            response.release_conn()

    def delete_object(self, key: str) -> None:
        self.client.remove_object(
            bucket_name=self.bucket,
            object_name=key,
        )
        logger.debug("delete_object", key=key)

    def delete_objects_with_prefix(self, prefix: str) -> int:
        keys = self.list_objects(prefix)
        for key in keys:
            self.delete_object(key)
        return len(keys)

    def list_objects(self, prefix: str) -> list[str]:
        objects = self.client.list_objects(
            self.bucket, prefix=prefix, recursive=True
        )
        return [obj.object_name for obj in objects if not obj.is_dir]

    def object_exists(self, key: str) -> bool:
        try:
            self.client.stat_object(self.bucket, key)
            return True
        except S3Error:
            return False
