from fastapi import APIRouter

from .stationarity_testing import stationary_testing_router

preliminary_diagnosis_router = APIRouter(prefix='/preliminary_diagnosis', tags=['Предварительная диагностика'])

preliminary_diagnosis_router.include_router(stationary_testing_router)
