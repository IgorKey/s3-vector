from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class CreateCollectionRequest(BaseModel):
    name: str = Field(min_length=1, max_length=256)
    dimension: int = Field(gt=0, le=65536)
    distance_metric: str = Field(default="cosine", pattern="^(cosine|l2|ip)$")
    index_type: str = Field(default="hnsw", pattern="^(hnsw)$")


class CollectionMetadata(BaseModel):
    collection_id: str
    name: str
    dimension: int
    distance_metric: str
    index_type: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CollectionInfo(BaseModel):
    collection_id: str
    name: str
    dimension: int
    distance_metric: str
    index_type: str
    vector_count: int
    created_at: datetime


class PutVectorRequest(BaseModel):
    id: str = Field(min_length=1, max_length=512)
    vector: list[float]
    metadata: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None


class PutVectorsRequest(BaseModel):
    vectors: list[PutVectorRequest] = Field(min_length=1, max_length=1000)


class VectorObject(BaseModel):
    id: str
    collection_id: str
    vector: list[float]
    metadata: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None


class SearchRequest(BaseModel):
    query_vector: list[float]
    top_k: int = Field(default=10, gt=0, le=1000)
    min_score: float | None = None
    include_payload: bool = False
    include_metadata: bool = True
    filter: dict[str, Any] | None = None


class SearchResult(BaseModel):
    id: str
    score: float
    metadata: dict[str, Any] | None = None
    payload: dict[str, Any] | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
