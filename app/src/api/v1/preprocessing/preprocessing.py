from dishka import FromDishka
from fastapi import APIRouter
from dishka.integrations.fastapi import inject_sync

from src.core.application.preprocessing.preprocess_scheme import PreprocessingRequest
from src.core.application.preprocessing.preprocessing_uc import PreprocessUC
from src.core.domain import Timeseries

preprocessing_router = APIRouter(prefix="/preprocessing", tags=["Предобработка ряда"])


@preprocessing_router.post("/")
@inject_sync
def preprocessing(
    request: PreprocessingRequest,
    preprocess_uc: FromDishka[PreprocessUC],
) -> Timeseries:
    return preprocess_uc.execute(request)
