from fastapi import APIRouter
from services.registry_service import promote_model

router = APIRouter()

@router.post("/promote")
def promote(model_version: str):

    result = promote_model(model_version)

    return {
        "success": True,
        "message": f"Model {model_version} promoted to Production"
    }