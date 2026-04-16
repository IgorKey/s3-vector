import structlog
from minio.error import S3Error

from src.index_engine import IndexEngine
from src.models import CollectionMetadata, CreateCollectionRequest
from src.s3_storage import S3Storage

logger = structlog.get_logger()


class CollectionManager:
    def __init__(self, storage: S3Storage):
        self.storage = storage
        self._indexes: dict[str, IndexEngine] = {}

    def _meta_key(self, collection_id: str) -> str:
        return f"collections/{collection_id}/meta.json"

    def _snapshot_key(self, collection_id: str) -> str:
        return f"collections/{collection_id}/index/snapshot.bin"

    def _vectors_prefix(self, collection_id: str) -> str:
        return f"collections/{collection_id}/vectors/"

    def create_collection(self, req: CreateCollectionRequest) -> CollectionMetadata:
        key = self._meta_key(req.name)
        if self.storage.object_exists(key):
            raise ValueError(f"Collection '{req.name}' already exists")

        meta = CollectionMetadata(
            collection_id=req.name,
            name=req.name,
            dimension=req.dimension,
            distance_metric=req.distance_metric,
            index_type=req.index_type,
        )
        self.storage.put_json_object(key, meta.model_dump(mode="json"))
        logger.info("collection_created", collection_id=meta.collection_id)
        return meta

    def get_collection(self, collection_id: str) -> CollectionMetadata:
        key = self._meta_key(collection_id)
        try:
            data = self.storage.get_json_object(key)
        except S3Error as e:
            raise KeyError(f"Collection '{collection_id}' not found") from e
        return CollectionMetadata(**data)

    def list_collections(self) -> list[CollectionMetadata]:
        keys = self.storage.list_objects("collections/")
        meta_keys = [k for k in keys if k.endswith("/meta.json")]
        result = []
        for key in meta_keys:
            data = self.storage.get_json_object(key)
            result.append(CollectionMetadata(**data))
        return result

    def delete_collection(self, collection_id: str) -> None:
        self.get_collection(collection_id)
        self._indexes.pop(collection_id, None)
        prefix = f"collections/{collection_id}/"
        self.storage.delete_objects_with_prefix(prefix)
        logger.info("collection_deleted", collection_id=collection_id)

    def get_or_create_index(self, meta: CollectionMetadata) -> IndexEngine:
        if meta.collection_id in self._indexes:
            return self._indexes[meta.collection_id]

        snapshot_key = self._snapshot_key(meta.collection_id)
        if self.storage.object_exists(snapshot_key):
            try:
                data = self.storage.get_bytes_object(snapshot_key)
                engine = IndexEngine.load_from_bytes(
                    data,
                    dimension=meta.dimension,
                    metric=meta.distance_metric,
                )
                self._indexes[meta.collection_id] = engine
                logger.info(
                    "index_loaded_from_snapshot",
                    collection_id=meta.collection_id,
                    count=engine.count(),
                )
                return engine
            except Exception:
                logger.warning(
                    "snapshot_load_failed",
                    collection_id=meta.collection_id,
                    exc_info=True,
                )

        engine = IndexEngine(
            dimension=meta.dimension,
            metric=meta.distance_metric,
        )
        self._rebuild_index_from_s3(meta.collection_id, engine)
        self._indexes[meta.collection_id] = engine
        return engine

    def _rebuild_index_from_s3(
        self, collection_id: str, engine: IndexEngine
    ) -> None:
        prefix = self._vectors_prefix(collection_id)
        keys = self.storage.list_objects(prefix)
        count = 0
        for key in keys:
            data = self.storage.get_json_object(key)
            engine.add(data["id"], data["vector"])
            count += 1
        logger.info(
            "index_rebuilt_from_s3",
            collection_id=collection_id,
            vector_count=count,
        )

    def save_snapshot(self, collection_id: str) -> None:
        if collection_id not in self._indexes:
            return
        engine = self._indexes[collection_id]
        data = engine.save_to_bytes()
        key = self._snapshot_key(collection_id)
        self.storage.put_bytes_object(key, data)
        logger.info(
            "snapshot_saved",
            collection_id=collection_id,
            size_bytes=len(data),
        )

    def save_all_snapshots(self) -> None:
        for collection_id in list(self._indexes):
            self.save_snapshot(collection_id)
