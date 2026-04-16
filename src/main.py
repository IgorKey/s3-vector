import asyncio
import logging
from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from src.collection_manager import CollectionManager
from src.config import settings
from src.routes.collections import create_collection_router
from src.routes.search import create_search_router
from src.routes.vectors import create_vector_router
from src.s3_storage import S3Storage
from src.vector_service import VectorService

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        logging.getLevelName(settings.log_level)
    ),
)

logger = structlog.get_logger()

_snapshot_task: asyncio.Task | None = None


async def _periodic_snapshot(cm: CollectionManager):
    """Periodically save index snapshots to S3."""
    while True:
        await asyncio.sleep(settings.snapshot_interval_seconds)
        try:
            cm.save_all_snapshots()
        except Exception:
            logger.error("snapshot_save_failed", exc_info=True)


def create_app() -> FastAPI:
    storage = S3Storage()
    collection_manager = CollectionManager(storage=storage)
    vector_service = VectorService(
        storage=storage, collection_manager=collection_manager
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        global _snapshot_task

        logger.info("starting_up", s3_endpoint=settings.s3_endpoint)

        collections = collection_manager.list_collections()
        for col in collections:
            collection_manager.get_or_create_index(col)
            vector_service.load_metadata_from_s3(col.collection_id)
        logger.info("loaded_collections", count=len(collections))

        _snapshot_task = asyncio.create_task(
            _periodic_snapshot(collection_manager)
        )

        yield

        _snapshot_task.cancel()
        collection_manager.save_all_snapshots()
        logger.info("shutdown_complete")

    application = FastAPI(
        title="S3 Vector Store",
        description="PoC Vector Store over S3-compatible Object Storage",
        version="0.1.0",
        lifespan=lifespan,
    )

    application.include_router(create_collection_router(collection_manager))
    application.include_router(create_vector_router(vector_service))
    application.include_router(create_search_router(vector_service))

    @application.get("/health")
    def health():
        return {"status": "ok"}

    @application.get("/stats")
    def stats():
        collections = collection_manager.list_collections()
        total_vectors = 0
        col_stats = []
        for col in collections:
            idx = collection_manager._indexes.get(col.collection_id)
            count = idx.count() if idx else 0
            total_vectors += count
            col_stats.append(
                {
                    "collection_id": col.collection_id,
                    "dimension": col.dimension,
                    "vector_count": count,
                }
            )

        return {
            "status": "ok",
            "total_collections": len(collections),
            "total_vectors": total_vectors,
            "collections": col_stats,
        }

    return application


app = create_app()

if __name__ == "__main__":
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
