from fastapi import APIRouter
from .building_model import fit_model_router
from .predict_series import model_predict_router
from .model_diagnosis import model_diagnosis_router
from .preliminary_diagnosis import preliminary_diagnosis_router
from .preprocessing import preprocessing_router
from .generating_series import generating_series_router
from .validate_series import validation_router

v1_router = APIRouter(prefix="/v1")
v1_router.include_router(fit_model_router)
v1_router.include_router(model_predict_router)
v1_router.include_router(preliminary_diagnosis_router)
v1_router.include_router(preprocessing_router)
v1_router.include_router(generating_series_router)
v1_router.include_router(model_diagnosis_router)
v1_router.include_router(validation_router)

__all__ = ("v1_router",)
