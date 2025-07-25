from fastapi import APIRouter
from .resid_analysis import resid_analysis_router


model_diagnosis_router = APIRouter(prefix='/model_diagnosis')

model_diagnosis_router.include_router(resid_analysis_router)
