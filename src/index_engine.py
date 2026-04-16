import tempfile
import os
from typing import ClassVar

import hnswlib
import numpy as np
import structlog

logger = structlog.get_logger()

METRIC_MAP = {
    "cosine": "cosine",
    "l2": "l2",
    "ip": "ip",
}


class IndexEngine:
    def __init__(
        self,
        dimension: int,
        metric: str = "cosine",
        max_elements: int = 100_000,
        ef_construction: int = 200,
        m: int = 16,
    ):
        self.dimension = dimension
        self.metric = metric
        self.max_elements = max_elements
        self._id_to_label: dict[str, int] = {}
        self._label_to_id: dict[int, str] = {}
        self._next_label = 0
        self._deleted_labels: set[int] = set()

        space = METRIC_MAP.get(metric, "cosine")
        self._index = hnswlib.Index(space=space, dim=dimension)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=ef_construction,
            M=m,
        )
        self._index.set_ef(50)

    def add(self, vector_id: str, vector: list[float]) -> None:
        vec = np.array(vector, dtype=np.float32)

        if vector_id in self._id_to_label:
            old_label = self._id_to_label[vector_id]
            self._index.mark_deleted(old_label)
            self._deleted_labels.add(old_label)
            del self._label_to_id[old_label]

        label = self._next_label
        self._next_label += 1

        self._maybe_resize(label + 1)
        self._index.add_items(vec.reshape(1, -1), np.array([label]))
        self._id_to_label[vector_id] = label
        self._label_to_id[label] = vector_id

    def _maybe_resize(self, needed: int) -> None:
        if needed > self.max_elements:
            new_max = max(needed, self.max_elements * 2)
            self._index.resize_index(new_max)
            self.max_elements = new_max

    def delete(self, vector_id: str) -> bool:
        if vector_id not in self._id_to_label:
            return False
        label = self._id_to_label.pop(vector_id)
        del self._label_to_id[label]
        self._index.mark_deleted(label)
        self._deleted_labels.add(label)
        return True

    def search(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        active_count = self.count()
        if active_count == 0:
            return []

        effective_k = min(top_k, active_count)
        vec = np.array(query_vector, dtype=np.float32).reshape(1, -1)
        labels, distances = self._index.knn_query(vec, k=effective_k)

        results = []
        for label, distance in zip(labels[0], distances[0]):
            vector_id = self._label_to_id.get(int(label))
            if vector_id is None:
                continue
            score = self._distance_to_score(float(distance))
            results.append((vector_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _distance_to_score(self, distance: float) -> float:
        if self.metric == "cosine":
            return 1.0 - distance
        elif self.metric == "l2":
            return 1.0 / (1.0 + distance)
        elif self.metric == "ip":
            return -distance
        return 1.0 - distance

    def count(self) -> int:
        return len(self._id_to_label)

    def save_to_bytes(self) -> bytes:
        import json

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "index.bin")
            self._index.save_index(index_path)
            with open(index_path, "rb") as f:
                index_data = f.read()

        mapping = {
            "id_to_label": self._id_to_label,
            "label_to_id": {str(k): v for k, v in self._label_to_id.items()},
            "next_label": self._next_label,
            "deleted_labels": list(self._deleted_labels),
        }
        mapping_data = json.dumps(mapping).encode("utf-8")

        length_bytes = len(mapping_data).to_bytes(4, "big")
        return length_bytes + mapping_data + index_data

    @classmethod
    def load_from_bytes(
        cls,
        data: bytes,
        dimension: int,
        metric: str = "cosine",
        max_elements: int = 100_000,
    ) -> "IndexEngine":
        import json

        mapping_len = int.from_bytes(data[:4], "big")
        mapping_data = json.loads(data[4 : 4 + mapping_len])
        index_data = data[4 + mapping_len :]

        engine = cls.__new__(cls)
        engine.dimension = dimension
        engine.metric = metric
        engine.max_elements = max_elements
        engine._id_to_label = mapping_data["id_to_label"]
        engine._label_to_id = {
            int(k): v for k, v in mapping_data["label_to_id"].items()
        }
        engine._next_label = mapping_data["next_label"]
        engine._deleted_labels = set(mapping_data["deleted_labels"])

        space = METRIC_MAP.get(metric, "cosine")
        engine._index = hnswlib.Index(space=space, dim=dimension)

        with tempfile.TemporaryDirectory() as tmpdir:
            index_path = os.path.join(tmpdir, "index.bin")
            with open(index_path, "wb") as f:
                f.write(index_data)
            engine._index.load_index(index_path, max_elements=max_elements)

        engine._index.set_ef(50)
        logger.info(
            "index_loaded_from_bytes",
            dimension=dimension,
            metric=metric,
            count=engine.count(),
        )
        return engine
