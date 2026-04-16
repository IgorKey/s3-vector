import pytest


@pytest.fixture
def sample_vector():
    """A 4-dimensional test vector."""
    return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def sample_metadata():
    return {"source": "test", "document_id": "doc-1"}
