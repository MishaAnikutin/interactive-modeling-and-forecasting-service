from fastapi import APIRouter, Response
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.building_model.errors.lstm import LstmFitValidationError, LstmPydanticValidationError
from src.core.application.building_model.errors.nhits import NhitsFitValidationError, NhitsPydanticValidationError
from src.core.application.building_model.schemas import NhitsFitRequest_V2, LstmFitRequest_V2, GruFitRequest_V2
from src.core.application.building_model.use_cases.models_v2 import FitNhitsUC_V2, FitLstmUC_V2, FitGruUC_V2

fit_model_router = APIRouter(prefix="/building_model", tags=["Построение модели"])

@fit_model_router.post(
    path="/nhits/fit",
    responses={
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
    request: NhitsFitRequest_V2,
    fit_nhits_uc: FromDishka[FitNhitsUC_V2]
) -> Response:
    archive_response = fit_nhits_uc.execute(request=request)
    return Response(
        content=archive_response,
        media_type="application/octet-stream",
    )


@fit_model_router.post(
    path="/lstm/fit",
    responses={
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
    request: LstmFitRequest_V2,
    fit_lstm_uc: FromDishka[FitLstmUC_V2]
) -> Response:
    archive_response = fit_lstm_uc.execute(request=request)
    return Response(
        content=archive_response,
        media_type="application/octet-stream",
    )


@fit_model_router.post(
    path="/gru/fit",
    responses={
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
    request: GruFitRequest_V2,
    fit_gru_uc: FromDishka[FitGruUC_V2]
) -> Response:
    archive_response = fit_gru_uc.execute(request=request)
    return Response(
        content=archive_response,
        media_type="application/octet-stream",
    )
