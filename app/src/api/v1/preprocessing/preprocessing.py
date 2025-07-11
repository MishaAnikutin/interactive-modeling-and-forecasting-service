from typing import List

from dishka import FromDishka
from fastapi import APIRouter
from dishka.integrations.fastapi import inject_sync

from src.core.application.preprocessing.preprocess_scheme import Preprocess
from src.core.application.preprocessing.preprocessing_uc import PreprocessUC
from src.core.domain import Timeseries

preprocessing_router = APIRouter(prefix="/stationary_testing", tags=["Анализ ряда на стационарность"])


@preprocessing_router.post("/stationary_testing")
@inject_sync
def preprocessing(
    ts: Timeseries,
    preprocess_list: List[Preprocess],
    preprocess_uc: FromDishka[PreprocessUC],
):
    pass
