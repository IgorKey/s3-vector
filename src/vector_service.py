import time
from typing import Any

import structlog

from src.collection_manager import CollectionManager
from src.models import (
    PutVectorRequest,
    SearchRequest,
    SearchResponse,
    SearchResult,
    VectorObject,
)
from src.s3_storage import S3Storage

logger = structlog.get_logger()


class VectorService:
    def __init__(self, storage: S3Storage, collection_manager: CollectionManager):
        self.storage = storage
        self.collection_manager = collection_manager
        self._metadata_store: dict[str, dict[str, dict[str, Any]]] = {}

    def _vector_key(self, collection_id: str, vector_id: str) -> str:
        return f"collections/{collection_id}/vectors/{vector_id}.json"

    def put_vector(self, collection_id: str, req: PutVectorRequest) -> VectorObject:
        meta = self.collection_manager.get_collection(collection_id)

        if len(req.vector) != meta.dimension:
            raise ValueError(
                f"Vector dimension {len(req.vector)} does not match "
                f"collection dimension {meta.dimension}"
            )

        obj = VectorObject(
            id=req.id,
            collection_id=collection_id,
            vector=req.vector,
            metadata=req.metadata,
            payload=req.payload,
        )

        key = self._vector_key(collection_id, req.id)
        self.storage.put_json_object(key, obj.model_dump(mode="json"))

        engine = self.collection_manager.get_or_create_index(meta)
        engine.add(req.id, req.vector)

        if collection_id not in self._metadata_store:
            self._metadata_store[collection_id] = {}
        if req.metadata:
            self._metadata_store[collection_id][req.id] = req.metadata

        logger.debug("vector_put", collection_id=collection_id, vector_id=req.id)
        return obj

    def put_vectors(
        self, collection_id: str, requests: list[PutVectorRequest]
    ) -> list[VectorObject]:
        results = []
        for req in requests:
            results.append(self.put_vector(collection_id, req))
        return results

    def get_vector(self, collection_id: str, vector_id: str) -> VectorObject:
        self.collection_manager.get_collection(collection_id)
        key = self._vector_key(collection_id, vector_id)
        data = self.storage.get_json_object(key)
        return VectorObject(**data)

    def delete_vector(self, collection_id: str, vector_id: str) -> None:
        meta = self.collection_manager.get_collection(collection_id)

        key = self._vector_key(collection_id, vector_id)
        self.storage.delete_object(key)

        engine = self.collection_manager.get_or_create_index(meta)
        engine.delete(vector_id)

        if collection_id in self._metadata_store:
            self._metadata_store[collection_id].pop(vector_id, None)

        logger.debug(
            "vector_deleted", collection_id=collection_id, vector_id=vector_id
        )

    def search_vectors(
        self, collection_id: str, req: SearchRequest
    ) -> SearchResponse:
        meta = self.collection_manager.get_collection(collection_id)

        if len(req.query_vector) != meta.dimension:
            raise ValueError(
                f"Query vector dimension {len(req.query_vector)} does not match "
                f"collection dimension {meta.dimension}"
            )

        start = time.monotonic()
        engine = self.collection_manager.get_or_create_index(meta)

        fetch_k = req.top_k * 3 if req.filter else req.top_k
        candidates = engine.search(req.query_vector, top_k=fetch_k)

        results = []
        col_metadata = self._metadata_store.get(collection_id, {})

        for vector_id, score in candidates:
            if req.min_score is not None and score < req.min_score:
                continue

            vec_meta = col_metadata.get(vector_id)

            if req.filter and vec_meta:
                if not self._matches_filter(vec_meta, req.filter):
                    continue
            elif req.filter and not vec_meta:
                continue

            result = SearchResult(id=vector_id, score=round(score, 6))

            if req.include_metadata and vec_meta:
                result.metadata = vec_meta

            if req.include_payload:
                try:
                    obj = self.get_vector(collection_id, vector_id)
                    result.payload = obj.payload
                    if req.include_metadata and obj.metadata:
                        result.metadata = obj.metadata
                except Exception:
                    pass

            results.append(result)
            if len(results) >= req.top_k:
                break

        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "search_completed",
            collection_id=collection_id,
            top_k=req.top_k,
            results=len(results),
            elapsed_ms=round(elapsed_ms, 2),
        )
        return SearchResponse(results=results)

    @staticmethod
    def _matches_filter(
        metadata: dict[str, Any], filter_: dict[str, Any]
    ) -> bool:
        for key, value in filter_.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True

    def load_metadata_from_s3(self, collection_id: str) -> None:
        prefix = f"collections/{collection_id}/vectors/"
        keys = self.storage.list_objects(prefix)
        self._metadata_store[collection_id] = {}
        for key in keys:
            data = self.storage.get_json_object(key)
            if data.get("metadata"):
                self._metadata_store[collection_id][data["id"]] = data["metadata"]
        logger.info(
            "metadata_loaded",
            collection_id=collection_id,
            count=len(self._metadata_store[collection_id]),
        )
