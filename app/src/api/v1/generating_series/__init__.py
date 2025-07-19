from fastapi import APIRouter
from .seasonal_decomposition import seasonal_decomposition_router


generating_series_router = APIRouter(prefix='/generating_series', tags=['Создание рядов'])

generating_series_router.include_router(seasonal_decomposition_router)
