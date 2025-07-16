from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync


seasonal_decomposition_router = APIRouter(
    prefix="/seasonal_decomposition",
    tags=["Сезонная декомпозиция"]
)


