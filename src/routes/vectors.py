from fastapi import APIRouter, HTTPException, Response

from src.models import PutVectorRequest, PutVectorsRequest, VectorObject
from src.vector_service import VectorService


def create_vector_router(vs: VectorService) -> APIRouter:
    router = APIRouter(prefix="/collections/{collection_id}", tags=["vectors"])

    @router.put("/vectors/{vector_id}")
    def put_vector(
        collection_id: str, vector_id: str, req: PutVectorRequest
    ) -> VectorObject:
        req.id = vector_id
        try:
            return vs.put_vector(collection_id, req)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError:
            raise HTTPException(status_code=404, detail="Collection not found")

    @router.get("/vectors/{vector_id}")
    def get_vector(collection_id: str, vector_id: str) -> VectorObject:
        try:
            return vs.get_vector(collection_id, vector_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Not found")

    @router.delete("/vectors/{vector_id}", status_code=204)
    def delete_vector(collection_id: str, vector_id: str) -> Response:
        try:
            vs.delete_vector(collection_id, vector_id)
            return Response(status_code=204)
        except KeyError:
            raise HTTPException(status_code=404, detail="Not found")

    @router.post("/vectors:batchPut")
    def batch_put_vectors(
        collection_id: str, req: PutVectorsRequest
    ) -> dict:
        try:
            results = vs.put_vectors(collection_id, req.vectors)
            return {"count": len(results), "status": "ok"}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except KeyError:
            raise HTTPException(status_code=404, detail="Collection not found")

    return router
