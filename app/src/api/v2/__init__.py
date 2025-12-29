from fastapi import APIRouter
from .building_model import fit_model_router
from .predict_series import model_predict_router

v2_router = APIRouter(prefix="/v2")
v2_router.include_router(fit_model_router)
v2_router.include_router(model_predict_router)

__all__ = ("v2_router",)
