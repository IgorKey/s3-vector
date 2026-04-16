import json
import io
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_minio_client():
    return MagicMock()


@pytest.fixture
def storage(mock_minio_client):
    from src.s3_storage import S3Storage

    s = S3Storage.__new__(S3Storage)
    s.client = mock_minio_client
    s.bucket = "test-bucket"
    return s


def test_put_json_object(storage, mock_minio_client):
    storage.put_json_object("collections/test/meta.json", {"name": "test"})
    mock_minio_client.put_object.assert_called_once()
    call_args = mock_minio_client.put_object.call_args
    assert call_args[1]["bucket_name"] == "test-bucket"
    assert call_args[1]["object_name"] == "collections/test/meta.json"


def test_get_json_object(storage, mock_minio_client):
    data = json.dumps({"name": "test"}).encode()
    response_mock = MagicMock()
    response_mock.read.return_value = data
    response_mock.close.return_value = None
    response_mock.release_conn.return_value = None
    mock_minio_client.get_object.return_value = response_mock

    result = storage.get_json_object("collections/test/meta.json")
    assert result == {"name": "test"}


def test_delete_object(storage, mock_minio_client):
    storage.delete_object("collections/test/vectors/v1.json")
    mock_minio_client.remove_object.assert_called_once_with(
        bucket_name="test-bucket",
        object_name="collections/test/vectors/v1.json",
    )


def test_list_objects_with_prefix(storage, mock_minio_client):
    obj1 = MagicMock()
    obj1.object_name = "collections/test/vectors/v1.json"
    obj1.is_dir = False
    obj2 = MagicMock()
    obj2.object_name = "collections/test/vectors/v2.json"
    obj2.is_dir = False
    mock_minio_client.list_objects.return_value = [obj1, obj2]

    result = storage.list_objects("collections/test/vectors/")
    assert result == [
        "collections/test/vectors/v1.json",
        "collections/test/vectors/v2.json",
    ]


def test_put_bytes_object(storage, mock_minio_client):
    storage.put_bytes_object("collections/test/index/snapshot.bin", b"binary-data")
    mock_minio_client.put_object.assert_called_once()


def test_get_bytes_object(storage, mock_minio_client):
    response_mock = MagicMock()
    response_mock.read.return_value = b"binary-data"
    response_mock.close.return_value = None
    response_mock.release_conn.return_value = None
    mock_minio_client.get_object.return_value = response_mock

    result = storage.get_bytes_object("collections/test/index/snapshot.bin")
    assert result == b"binary-data"


def test_object_exists_true(storage, mock_minio_client):
    mock_minio_client.stat_object.return_value = MagicMock()
    assert storage.object_exists("collections/test/meta.json") is True


def test_object_exists_false(storage, mock_minio_client):
    from minio.error import S3Error

    mock_minio_client.stat_object.side_effect = S3Error(
        "NoSuchKey", "Not found", "", "", "", ""
    )
    assert storage.object_exists("collections/test/meta.json") is False
