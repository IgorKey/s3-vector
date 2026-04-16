from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    s3_endpoint: str = "localhost:9000"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "vector-store"
    s3_use_ssl: bool = False

    snapshot_interval_seconds: int = 60
    log_level: str = "INFO"

    model_config = {"env_prefix": "S3V_"}


settings = Settings()
