from fastapi import APIRouter, HTTPException

from src.models import SearchRequest, SearchResponse
from src.vector_service import VectorService


def create_search_router(vs: VectorService) -> APIRouter:
    router = APIRouter(prefix="/collections/{collection_id}", tags=["search"])

    @router.post("/search")
    def search_vectors(
        collection_id: str, req: SearchRequest
    ) -> SearchResponse:
        try:
            return vs.search_vectors(collection_id, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError:
            raise HTTPException(status_code=404, detail="Collection not found")

    return router
