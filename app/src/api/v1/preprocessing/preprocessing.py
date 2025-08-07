from dishka import FromDishka
from fastapi import APIRouter, HTTPException
from dishka.integrations.fastapi import inject_sync

from src.core.domain import Timeseries
from src.core.application.preprocessing.preprocess_scheme import (
    PreprocessingRequest,
    InversePreprocessingRequest,
    PreprocessingResponse
)
from src.core.application.preprocessing import PreprocessUC, InversePreprocessUC

preprocessing_router = APIRouter(prefix="/preprocessing", tags=["Предобработка ряда"])


@preprocessing_router.post("/apply", summary='Предобработать ряд')
@inject_sync
def preprocessing(
    request: PreprocessingRequest,
    preprocess_uc: FromDishka[PreprocessUC],
) -> PreprocessingResponse:
    try:
        return preprocess_uc.execute(request)
    except ValueError as detail:
        raise HTTPException(status_code=400, detail=str(detail))


@preprocessing_router.post("/inverse", summary='Вернуть ряд к исходному после предобработок')
@inject_sync
def inverse_preprocessing(
    request: InversePreprocessingRequest,
    inverse_preprocess_uc: FromDishka[InversePreprocessUC],
) -> Timeseries:
    try:
        return inverse_preprocess_uc.execute(request)
    except ValueError as detail:
        raise HTTPException(status_code=400, detail=str(detail))
