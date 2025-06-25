from fastapi import APIRouter
from .building_model import fit_model_router, params_router
from .preliminary_diagnosis import stationary_testing_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(fit_model_router)
v1_router.include_router(params_router)
v1_router.include_router(stationary_testing_router)


__all__ = ("v1_router",)
