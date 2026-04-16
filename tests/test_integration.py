"""
Integration test: runs full vector lifecycle without Docker.
Uses mock S3 storage (in-memory dict) to avoid external dependencies.
"""

import pytest
import json


class InMemoryStorage:
    """Fake S3 storage backed by a dict, for integration testing."""

    def __init__(self):
        self._objects: dict[str, bytes] = {}

    def put_json_object(self, key: str, data: dict) -> None:
        self._objects[key] = json.dumps(data, default=str).encode()

    def get_json_object(self, key: str) -> dict:
        if key not in self._objects:
            from minio.error import S3Error
            raise S3Error("NoSuchKey", "Not found", "", "", "", "")
        return json.loads(self._objects[key])

    def put_bytes_object(self, key: str, data: bytes) -> None:
        self._objects[key] = data

    def get_bytes_object(self, key: str) -> bytes:
        return self._objects[key]

    def delete_object(self, key: str) -> None:
        self._objects.pop(key, None)

    def delete_objects_with_prefix(self, prefix: str) -> int:
        keys = [k for k in self._objects if k.startswith(prefix)]
        for k in keys:
            del self._objects[k]
        return len(keys)

    def list_objects(self, prefix: str) -> list[str]:
        return [k for k in sorted(self._objects) if k.startswith(prefix)]

    def object_exists(self, key: str) -> bool:
        return key in self._objects


def test_full_lifecycle():
    from src.collection_manager import CollectionManager
    from src.vector_service import VectorService
    from src.models import (
        CreateCollectionRequest,
        PutVectorRequest,
        SearchRequest,
    )

    storage = InMemoryStorage()
    cm = CollectionManager(storage=storage)
    vs = VectorService(storage=storage, collection_manager=cm)

    # Create collection
    col = cm.create_collection(
        CreateCollectionRequest(name="test", dimension=4, distance_metric="cosine")
    )
    assert col.collection_id == "test"

    # Put vectors
    vs.put_vector(
        "test",
        PutVectorRequest(id="v1", vector=[1.0, 0.0, 0.0, 0.0], metadata={"tag": "a"}),
    )
    vs.put_vector(
        "test",
        PutVectorRequest(id="v2", vector=[0.0, 1.0, 0.0, 0.0], metadata={"tag": "b"}),
    )
    vs.put_vector(
        "test",
        PutVectorRequest(
            id="v3",
            vector=[0.9, 0.1, 0.0, 0.0],
            metadata={"tag": "a"},
            payload={"text": "hello"},
        ),
    )

    # Search
    results = vs.search_vectors(
        "test",
        SearchRequest(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2),
    )
    assert len(results.results) == 2
    assert results.results[0].id == "v1"

    # Search with filter
    results = vs.search_vectors(
        "test",
        SearchRequest(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=10,
            filter={"tag": "b"},
        ),
    )
    assert len(results.results) == 1
    assert results.results[0].id == "v2"

    # Search with min_score
    results = vs.search_vectors(
        "test",
        SearchRequest(
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=10,
            min_score=0.999,
        ),
    )
    assert len(results.results) == 1
    assert results.results[0].id == "v1"

    # Get vector
    obj = vs.get_vector("test", "v1")
    assert obj.vector == [1.0, 0.0, 0.0, 0.0]

    # Delete vector
    vs.delete_vector("test", "v1")
    results = vs.search_vectors(
        "test",
        SearchRequest(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=10),
    )
    assert all(r.id != "v1" for r in results.results)

    # Snapshot + recovery
    cm.save_snapshot("test")
    cm2 = CollectionManager(storage=storage)
    vs2 = VectorService(storage=storage, collection_manager=cm2)
    meta = cm2.get_collection("test")
    cm2.get_or_create_index(meta)
    vs2.load_metadata_from_s3("test")

    results = vs2.search_vectors(
        "test",
        SearchRequest(query_vector=[0.0, 1.0, 0.0, 0.0], top_k=1),
    )
    assert results.results[0].id == "v2"


def test_batch_put():
    from src.collection_manager import CollectionManager
    from src.vector_service import VectorService
    from src.models import CreateCollectionRequest, PutVectorRequest, SearchRequest

    storage = InMemoryStorage()
    cm = CollectionManager(storage=storage)
    vs = VectorService(storage=storage, collection_manager=cm)

    cm.create_collection(
        CreateCollectionRequest(name="batch-test", dimension=4)
    )

    vectors = [
        PutVectorRequest(id=f"v{i}", vector=[float(i == j) for j in range(4)])
        for i in range(4)
    ]
    vs.put_vectors("batch-test", vectors)

    results = vs.search_vectors(
        "batch-test",
        SearchRequest(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=1),
    )
    assert results.results[0].id == "v0"
