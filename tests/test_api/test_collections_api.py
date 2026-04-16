import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


@pytest.fixture
def mock_deps():
    from src.collection_manager import CollectionManager
    from src.vector_service import VectorService
    from src.s3_storage import S3Storage

    storage = MagicMock(spec=S3Storage)
    cm = MagicMock(spec=CollectionManager)
    vs = MagicMock(spec=VectorService)
    return storage, cm, vs


@pytest.fixture
def client(mock_deps):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.routes.collections import create_collection_router

    storage, cm, vs = mock_deps
    app = FastAPI()
    app.include_router(create_collection_router(cm))
    return TestClient(app), cm


def test_create_collection(client):
    test_client, cm = client
    from src.models import CollectionMetadata

    cm.create_collection.return_value = CollectionMetadata(
        collection_id="test-col",
        name="test-col",
        dimension=1536,
        distance_metric="cosine",
        index_type="hnsw",
    )

    resp = test_client.post(
        "/collections",
        json={"name": "test-col", "dimension": 1536},
    )
    assert resp.status_code == 201
    data = resp.json()
    assert data["collection_id"] == "test-col"


def test_create_collection_already_exists(client):
    test_client, cm = client
    cm.create_collection.side_effect = ValueError("already exists")

    resp = test_client.post(
        "/collections",
        json={"name": "test-col", "dimension": 1536},
    )
    assert resp.status_code == 409


def test_get_collection(client):
    test_client, cm = client
    from src.models import CollectionMetadata

    cm.get_collection.return_value = CollectionMetadata(
        collection_id="test-col",
        name="test-col",
        dimension=1536,
        distance_metric="cosine",
        index_type="hnsw",
    )

    resp = test_client.get("/collections/test-col")
    assert resp.status_code == 200
    assert resp.json()["collection_id"] == "test-col"


def test_get_collection_not_found(client):
    test_client, cm = client
    cm.get_collection.side_effect = KeyError("not found")

    resp = test_client.get("/collections/nonexistent")
    assert resp.status_code == 404


def test_list_collections(client):
    test_client, cm = client
    from src.models import CollectionMetadata

    cm.list_collections.return_value = [
        CollectionMetadata(
            collection_id="col-1",
            name="col-1",
            dimension=4,
            distance_metric="cosine",
            index_type="hnsw",
        )
    ]

    resp = test_client.get("/collections")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1


def test_delete_collection(client):
    test_client, cm = client

    resp = test_client.delete("/collections/test-col")
    assert resp.status_code == 204
    cm.delete_collection.assert_called_once_with("test-col")
