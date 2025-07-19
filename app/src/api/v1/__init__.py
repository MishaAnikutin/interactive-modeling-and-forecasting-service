from fastapi import APIRouter
from .building_model import fit_model_router, params_router
from .preliminary_diagnosis import preliminary_diagnosis_router
from .preprocessing import preprocessing_router
from .generating_series import generating_series_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(fit_model_router)
v1_router.include_router(params_router)
v1_router.include_router(preliminary_diagnosis_router)
v1_router.include_router(preprocessing_router)
v1_router.include_router(generating_series_router)

__all__ = ("v1_router",)
