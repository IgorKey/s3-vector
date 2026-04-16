import pytest
from datetime import datetime, timezone


def test_create_collection_request_defaults():
    from src.models import CreateCollectionRequest

    req = CreateCollectionRequest(name="test", dimension=1536)
    assert req.distance_metric == "cosine"
    assert req.index_type == "hnsw"


def test_create_collection_request_validation():
    from src.models import CreateCollectionRequest

    with pytest.raises(Exception):
        CreateCollectionRequest(name="", dimension=0)


def test_collection_metadata():
    from src.models import CollectionMetadata

    meta = CollectionMetadata(
        collection_id="test-col",
        name="test",
        dimension=1536,
        distance_metric="cosine",
        index_type="hnsw",
    )
    assert meta.collection_id == "test-col"
    assert isinstance(meta.created_at, datetime)


def test_put_vector_request():
    from src.models import PutVectorRequest

    req = PutVectorRequest(
        id="vec-1",
        vector=[0.1, 0.2, 0.3],
        metadata={"source": "test"},
        payload={"text": "hello"},
    )
    assert req.id == "vec-1"
    assert len(req.vector) == 3


def test_vector_object():
    from src.models import VectorObject

    obj = VectorObject(
        id="vec-1",
        collection_id="col-1",
        vector=[0.1, 0.2],
        metadata={"source": "test"},
    )
    assert obj.collection_id == "col-1"
    assert isinstance(obj.created_at, datetime)


def test_search_request_defaults():
    from src.models import SearchRequest

    req = SearchRequest(query_vector=[0.1, 0.2], top_k=5)
    assert req.include_payload is False
    assert req.include_metadata is True
    assert req.min_score is None
    assert req.filter is None


def test_search_result():
    from src.models import SearchResult, SearchResponse

    result = SearchResult(id="vec-1", score=0.95)
    resp = SearchResponse(results=[result])
    assert len(resp.results) == 1
    assert resp.results[0].score == 0.95
