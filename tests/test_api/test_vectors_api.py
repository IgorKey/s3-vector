import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone


@pytest.fixture
def mock_deps():
    from src.collection_manager import CollectionManager
    from src.vector_service import VectorService

    cm = MagicMock(spec=CollectionManager)
    vs = MagicMock(spec=VectorService)
    return cm, vs


@pytest.fixture
def client(mock_deps):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.routes.vectors import create_vector_router

    cm, vs = mock_deps
    app = FastAPI()
    app.include_router(create_vector_router(vs))
    return TestClient(app), vs


def test_put_vector(client):
    test_client, vs = client
    from src.models import VectorObject

    vs.put_vector.return_value = VectorObject(
        id="vec-1",
        collection_id="col-1",
        vector=[0.1, 0.2, 0.3],
    )

    resp = test_client.put(
        "/collections/col-1/vectors/vec-1",
        json={"id": "vec-1", "vector": [0.1, 0.2, 0.3]},
    )
    assert resp.status_code == 200
    assert resp.json()["id"] == "vec-1"


def test_put_vector_dimension_mismatch(client):
    test_client, vs = client
    vs.put_vector.side_effect = ValueError("dimension mismatch")

    resp = test_client.put(
        "/collections/col-1/vectors/vec-1",
        json={"id": "vec-1", "vector": [0.1, 0.2]},
    )
    assert resp.status_code == 400


def test_get_vector(client):
    test_client, vs = client
    from src.models import VectorObject

    vs.get_vector.return_value = VectorObject(
        id="vec-1",
        collection_id="col-1",
        vector=[0.1, 0.2, 0.3],
    )

    resp = test_client.get("/collections/col-1/vectors/vec-1")
    assert resp.status_code == 200
    assert resp.json()["id"] == "vec-1"


def test_delete_vector(client):
    test_client, vs = client

    resp = test_client.delete("/collections/col-1/vectors/vec-1")
    assert resp.status_code == 204


def test_batch_put_vectors(client):
    test_client, vs = client
    from src.models import VectorObject

    vs.put_vectors.return_value = [
        VectorObject(id="v1", collection_id="col-1", vector=[0.1]),
        VectorObject(id="v2", collection_id="col-1", vector=[0.2]),
    ]

    resp = test_client.post(
        "/collections/col-1/vectors:batchPut",
        json={
            "vectors": [
                {"id": "v1", "vector": [0.1]},
                {"id": "v2", "vector": [0.2]},
            ]
        },
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 2
