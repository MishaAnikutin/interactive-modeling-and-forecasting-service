from fastapi import APIRouter, HTTPException
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.building_model.schemas.arimax import ArimaxFitRequest, ArimaxFitResult
from src.core.application.building_model.schemas.errors import PydanticValidationError, FitValidationError
from src.core.application.building_model.schemas.lstm import LstmFitRequest, LstmFitResult
from src.core.application.building_model.schemas.nhits import NhitsFitRequest, NhitsFitResult
from src.core.application.building_model.use_cases.fit_arimax import FitArimaxUC
from src.core.application.building_model.use_cases.fit_lstm import FitLstmUC
from src.core.application.building_model.use_cases.fit_nhits import FitNhitsUC

fit_model_router = APIRouter(prefix="/building_model", tags=["Построение модели"])


@fit_model_router.post("/arimax/fit")
@inject_sync
def fit_arimax(
    request: ArimaxFitRequest,
    fit_arimax_uc: FromDishka[FitArimaxUC]
) -> ArimaxFitResult:
    return fit_arimax_uc.execute(request=request)


@fit_model_router.post(
    path="/nhits/fit",
    responses={
        200: {"model": NhitsFitResult},
        400: {
            "model": FitValidationError,
            "description": "Ошибка валидации во время обучения",
        },
        422: {
            "model": PydanticValidationError,
            "description": "Ошибка валидации запроса",
        },
    }
)
@inject_sync
def fit_nhits(
    request: NhitsFitRequest,
    fit_nhits_uc: FromDishka[FitNhitsUC]
) -> NhitsFitResult:
    return fit_nhits_uc.execute(request=request)


@fit_model_router.post("/lstm/fit")
@inject_sync
def fit_lstm(
    request: LstmFitRequest,
    fit_lstm_uc: FromDishka[FitLstmUC]
) -> LstmFitResult:
    return fit_lstm_uc.execute(request=request)