from fastapi import APIRouter, HTTPException
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.building_model.errors.arimax import ArimaxFitValidationError
from src.core.application.building_model.errors.lstm import LstmFitValidationError, LstmPydanticValidationError
from src.core.application.building_model.errors.nhits import NhitsFitValidationError, NhitsPydanticValidationError
from src.core.application.building_model.schemas.arimax import ArimaxFitRequest, ArimaxFitResult
from src.core.application.building_model.schemas.gru import GruFitRequest, GruFitResult
from src.core.application.building_model.schemas.lstm import LstmFitRequest, LstmFitResult
from src.core.application.building_model.schemas.nhits import NhitsFitRequest, NhitsFitResult
from src.core.application.building_model.use_cases.fit_arimax import FitArimaxUC
from src.core.application.building_model.use_cases.fit_gru import FitGruUC
from src.core.application.building_model.use_cases.fit_lstm import FitLstmUC
from src.core.application.building_model.use_cases.fit_nhits import FitNhitsUC

fit_model_router = APIRouter(prefix="/building_model", tags=["Построение модели"])


@fit_model_router.post(
    path="/arimax/fit",
    responses={
        200: {
            "model": ArimaxFitResult,
            "description": "Успешный ответ"
        },
        400: {
            "model": ArimaxFitValidationError,
            "description": "Ошибка валидации во время обучения",
        },
    }
)
@inject_sync
def fit_arimax(
    request: ArimaxFitRequest,
    fit_arimax_uc: FromDishka[FitArimaxUC]
) -> ArimaxFitResult:
    return fit_arimax_uc.execute(request=request)


@fit_model_router.post(
    path="/nhits/fit",
    responses={
        200: {
            "model": NhitsFitResult,
            "description": "Успешный ответ"
        },
        400: {
            "model": NhitsFitValidationError,
            "description": "Ошибка валидации во время обучения",
        },
        422: {
            "model": NhitsPydanticValidationError,
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


@fit_model_router.post(
    path="/lstm/fit",
    responses={
        200: {
            "model": LstmFitResult,
            "description": "Успешный ответ"
        },
        400: {
            "model": LstmFitValidationError,
            "description": "Ошибка валидации во время обучения",
        },
        422: {
            "model": LstmPydanticValidationError,
            "description": "Ошибка валидации запроса",
        },
    }
)
@inject_sync
def fit_lstm(
    request: LstmFitRequest,
    fit_lstm_uc: FromDishka[FitLstmUC]
) -> LstmFitResult:
    return fit_lstm_uc.execute(request=request)


@fit_model_router.post(
    path="/gru/fit",
    responses={
        200: {
            "model": GruFitResult,
            "description": "Успешный ответ"
        },
        400: {
            "model": LstmFitValidationError,
            "description": "Ошибка валидации во время обучения",
        },
        422: {
            "model": LstmPydanticValidationError,
            "description": "Ошибка валидации запроса",
        },
    }
)
@inject_sync
def fit_gru(
    request: GruFitRequest,
    fit_gru_uc: FromDishka[FitGruUC]
) -> GruFitResult:
    return fit_gru_uc.execute(request=request)
