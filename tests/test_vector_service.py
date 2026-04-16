import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


@pytest.fixture
def mock_storage():
    return MagicMock()


@pytest.fixture
def mock_collection_manager():
    return MagicMock()


@pytest.fixture
def service(mock_storage, mock_collection_manager):
    from src.vector_service import VectorService

    return VectorService(
        storage=mock_storage,
        collection_manager=mock_collection_manager,
    )


def test_put_vector(service, mock_storage, mock_collection_manager):
    from src.models import PutVectorRequest, CollectionMetadata
    from src.index_engine import IndexEngine

    meta = CollectionMetadata(
        collection_id="col-1",
        name="col-1",
        dimension=4,
        distance_metric="cosine",
        index_type="hnsw",
    )
    mock_collection_manager.get_collection.return_value = meta

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    mock_collection_manager.get_or_create_index.return_value = engine

    req = PutVectorRequest(
        id="vec-1",
        vector=[1.0, 0.0, 0.0, 0.0],
        metadata={"source": "test"},
    )
    service.put_vector("col-1", req)

    mock_storage.put_json_object.assert_called_once()
    assert engine.count() == 1


def test_get_vector(service, mock_storage, mock_collection_manager):
    from src.models import CollectionMetadata

    meta = CollectionMetadata(
        collection_id="col-1",
        name="col-1",
        dimension=4,
        distance_metric="cosine",
        index_type="hnsw",
    )
    mock_collection_manager.get_collection.return_value = meta
    mock_storage.get_json_object.return_value = {
        "id": "vec-1",
        "collection_id": "col-1",
        "vector": [1.0, 0.0, 0.0, 0.0],
        "metadata": {"source": "test"},
        "created_at": "2026-04-16T10:00:00+00:00",
    }

    result = service.get_vector("col-1", "vec-1")
    assert result.id == "vec-1"


def test_delete_vector(service, mock_storage, mock_collection_manager):
    from src.models import CollectionMetadata
    from src.index_engine import IndexEngine

    meta = CollectionMetadata(
        collection_id="col-1",
        name="col-1",
        dimension=4,
        distance_metric="cosine",
        index_type="hnsw",
    )
    mock_collection_manager.get_collection.return_value = meta

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    engine.add("vec-1", [1.0, 0.0, 0.0, 0.0])
    mock_collection_manager.get_or_create_index.return_value = engine

    service.delete_vector("col-1", "vec-1")
    mock_storage.delete_object.assert_called_once()
    assert engine.count() == 0


def test_search_vectors(service, mock_storage, mock_collection_manager):
    from src.models import CollectionMetadata, SearchRequest
    from src.index_engine import IndexEngine

    meta = CollectionMetadata(
        collection_id="col-1",
        name="col-1",
        dimension=4,
        distance_metric="cosine",
        index_type="hnsw",
    )
    mock_collection_manager.get_collection.return_value = meta

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    engine.add("vec-1", [1.0, 0.0, 0.0, 0.0])
    engine.add("vec-2", [0.0, 1.0, 0.0, 0.0])
    mock_collection_manager.get_or_create_index.return_value = engine

    req = SearchRequest(query_vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
    results = service.search_vectors("col-1", req)

    assert len(results.results) == 2
    assert results.results[0].id == "vec-1"


def test_search_with_min_score(service, mock_storage, mock_collection_manager):
    from src.models import CollectionMetadata, SearchRequest
    from src.index_engine import IndexEngine

    meta = CollectionMetadata(
        collection_id="col-1",
        name="col-1",
        dimension=4,
        distance_metric="cosine",
        index_type="hnsw",
    )
    mock_collection_manager.get_collection.return_value = meta

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    engine.add("vec-1", [1.0, 0.0, 0.0, 0.0])
    engine.add("vec-2", [0.0, 1.0, 0.0, 0.0])
    mock_collection_manager.get_or_create_index.return_value = engine

    req = SearchRequest(
        query_vector=[1.0, 0.0, 0.0, 0.0], top_k=10, min_score=0.99
    )
    results = service.search_vectors("col-1", req)

    assert len(results.results) == 1
    assert results.results[0].id == "vec-1"


def test_search_with_metadata_filter(service, mock_storage, mock_collection_manager):
    from src.models import CollectionMetadata, SearchRequest
    from src.index_engine import IndexEngine

    meta = CollectionMetadata(
        collection_id="col-1",
        name="col-1",
        dimension=4,
        distance_metric="cosine",
        index_type="hnsw",
    )
    mock_collection_manager.get_collection.return_value = meta

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    engine.add("vec-1", [1.0, 0.0, 0.0, 0.0])
    engine.add("vec-2", [0.9, 0.1, 0.0, 0.0])
    mock_collection_manager.get_or_create_index.return_value = engine

    service._metadata_store["col-1"] = {
        "vec-1": {"source": "confluence"},
        "vec-2": {"source": "jira"},
    }

    req = SearchRequest(
        query_vector=[1.0, 0.0, 0.0, 0.0],
        top_k=10,
        filter={"source": "jira"},
    )
    results = service.search_vectors("col-1", req)

    assert len(results.results) == 1
    assert results.results[0].id == "vec-2"
