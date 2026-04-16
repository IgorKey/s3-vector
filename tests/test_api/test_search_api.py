import pytest
from unittest.mock import MagicMock


@pytest.fixture
def client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.routes.search import create_search_router
    from src.vector_service import VectorService

    vs = MagicMock(spec=VectorService)
    app = FastAPI()
    app.include_router(create_search_router(vs))
    return TestClient(app), vs


def test_search_vectors(client):
    test_client, vs = client
    from src.models import SearchResponse, SearchResult

    vs.search_vectors.return_value = SearchResponse(
        results=[SearchResult(id="vec-1", score=0.95)]
    )

    resp = test_client.post(
        "/collections/col-1/search",
        json={"query_vector": [0.1, 0.2, 0.3], "top_k": 5},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) == 1
    assert data["results"][0]["id"] == "vec-1"


def test_search_dimension_mismatch(client):
    test_client, vs = client
    vs.search_vectors.side_effect = ValueError("dimension mismatch")

    resp = test_client.post(
        "/collections/col-1/search",
        json={"query_vector": [0.1], "top_k": 5},
    )
    assert resp.status_code == 400


def test_search_collection_not_found(client):
    test_client, vs = client
    vs.search_vectors.side_effect = KeyError("not found")

    resp = test_client.post(
        "/collections/nonexistent/search",
        json={"query_vector": [0.1], "top_k": 5},
    )
    assert resp.status_code == 404
