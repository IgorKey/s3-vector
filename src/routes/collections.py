from fastapi import APIRouter, HTTPException, Response

from src.collection_manager import CollectionManager
from src.models import CreateCollectionRequest, CollectionMetadata


def create_collection_router(cm: CollectionManager) -> APIRouter:
    router = APIRouter(prefix="/collections", tags=["collections"])

    @router.post("", status_code=201)
    def create_collection(req: CreateCollectionRequest) -> dict:
        try:
            meta = cm.create_collection(req)
            return {"collection_id": meta.collection_id, "status": "created"}
        except ValueError as e:
            raise HTTPException(status_code=409, detail=str(e))

    @router.get("")
    def list_collections() -> list[CollectionMetadata]:
        return cm.list_collections()

    @router.get("/{collection_id}")
    def get_collection(collection_id: str) -> CollectionMetadata:
        try:
            return cm.get_collection(collection_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Collection not found")

    @router.delete("/{collection_id}", status_code=204)
    def delete_collection(collection_id: str) -> Response:
        try:
            cm.delete_collection(collection_id)
            return Response(status_code=204)
        except KeyError:
            raise HTTPException(status_code=404, detail="Collection not found")

    return router
