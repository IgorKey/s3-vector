import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone


@pytest.fixture
def mock_storage():
    return MagicMock()


@pytest.fixture
def manager(mock_storage):
    from src.collection_manager import CollectionManager

    return CollectionManager(storage=mock_storage)


def test_create_collection(manager, mock_storage):
    from src.models import CreateCollectionRequest

    mock_storage.object_exists.return_value = False

    req = CreateCollectionRequest(name="test-col", dimension=4)
    result = manager.create_collection(req)

    assert result.collection_id == "test-col"
    assert result.dimension == 4
    mock_storage.put_json_object.assert_called_once()


def test_create_collection_already_exists(manager, mock_storage):
    from src.models import CreateCollectionRequest

    mock_storage.object_exists.return_value = True

    req = CreateCollectionRequest(name="test-col", dimension=4)
    with pytest.raises(ValueError, match="already exists"):
        manager.create_collection(req)


def test_get_collection(manager, mock_storage):
    mock_storage.get_json_object.return_value = {
        "collection_id": "test-col",
        "name": "test-col",
        "dimension": 4,
        "distance_metric": "cosine",
        "index_type": "hnsw",
        "created_at": "2026-04-16T10:00:00+00:00",
    }

    result = manager.get_collection("test-col")
    assert result.collection_id == "test-col"


def test_list_collections(manager, mock_storage):
    mock_storage.list_objects.return_value = [
        "collections/col-1/meta.json",
        "collections/col-2/meta.json",
    ]
    mock_storage.get_json_object.side_effect = [
        {
            "collection_id": "col-1",
            "name": "col-1",
            "dimension": 4,
            "distance_metric": "cosine",
            "index_type": "hnsw",
            "created_at": "2026-04-16T10:00:00+00:00",
        },
        {
            "collection_id": "col-2",
            "name": "col-2",
            "dimension": 8,
            "distance_metric": "l2",
            "index_type": "hnsw",
            "created_at": "2026-04-16T10:00:00+00:00",
        },
    ]

    result = manager.list_collections()
    assert len(result) == 2


def test_delete_collection(manager, mock_storage):
    mock_storage.get_json_object.return_value = {
        "collection_id": "test-col",
        "name": "test-col",
        "dimension": 4,
        "distance_metric": "cosine",
        "index_type": "hnsw",
        "created_at": "2026-04-16T10:00:00+00:00",
    }
    mock_storage.delete_objects_with_prefix.return_value = 5

    manager.delete_collection("test-col")
    mock_storage.delete_objects_with_prefix.assert_called_once_with(
        "collections/test-col/"
    )


def test_get_index_creates_on_first_access(manager, mock_storage):
    from src.models import CollectionMetadata

    meta = CollectionMetadata(
        collection_id="test-col",
        name="test-col",
        dimension=4,
        distance_metric="cosine",
        index_type="hnsw",
    )
    mock_storage.object_exists.return_value = False

    index = manager.get_or_create_index(meta)
    assert index is not None
    assert index.count() == 0
