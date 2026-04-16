import tempfile
import os
import numpy as np
import pytest


def test_create_index():
    from src.index_engine import IndexEngine

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    assert engine.count() == 0


def test_add_and_search():
    from src.index_engine import IndexEngine

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    engine.add("vec-1", [1.0, 0.0, 0.0, 0.0])
    engine.add("vec-2", [0.0, 1.0, 0.0, 0.0])
    engine.add("vec-3", [0.9, 0.1, 0.0, 0.0])

    results = engine.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0][0] == "vec-1"
    assert 0 <= results[0][1] <= 1.0
    assert 0 <= results[1][1] <= 1.0


def test_add_duplicate_id_updates():
    from src.index_engine import IndexEngine

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    engine.add("vec-1", [1.0, 0.0, 0.0, 0.0])
    engine.add("vec-1", [0.0, 1.0, 0.0, 0.0])

    results = engine.search([0.0, 1.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == "vec-1"


def test_delete():
    from src.index_engine import IndexEngine

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    engine.add("vec-1", [1.0, 0.0, 0.0, 0.0])
    engine.add("vec-2", [0.0, 1.0, 0.0, 0.0])
    engine.delete("vec-1")

    results = engine.search([1.0, 0.0, 0.0, 0.0], top_k=2)
    ids = [r[0] for r in results]
    assert "vec-1" not in ids


def test_save_and_load():
    from src.index_engine import IndexEngine

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    engine.add("vec-1", [1.0, 0.0, 0.0, 0.0])
    engine.add("vec-2", [0.0, 1.0, 0.0, 0.0])

    data = engine.save_to_bytes()
    assert isinstance(data, bytes)
    assert len(data) > 0

    engine2 = IndexEngine.load_from_bytes(
        data, dimension=4, metric="cosine", max_elements=100
    )
    assert engine2.count() == 2

    results = engine2.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == "vec-1"


def test_search_empty_index():
    from src.index_engine import IndexEngine

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    results = engine.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_count():
    from src.index_engine import IndexEngine

    engine = IndexEngine(dimension=4, metric="cosine", max_elements=100)
    assert engine.count() == 0
    engine.add("a", [1.0, 0.0, 0.0, 0.0])
    assert engine.count() == 1
    engine.add("b", [0.0, 1.0, 0.0, 0.0])
    assert engine.count() == 2


def test_l2_metric():
    from src.index_engine import IndexEngine

    engine = IndexEngine(dimension=4, metric="l2", max_elements=100)
    engine.add("vec-1", [1.0, 0.0, 0.0, 0.0])
    engine.add("vec-2", [0.0, 1.0, 0.0, 0.0])

    results = engine.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == "vec-1"
