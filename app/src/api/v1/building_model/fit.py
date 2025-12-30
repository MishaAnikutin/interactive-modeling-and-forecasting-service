from fastapi import APIRouter, Response, HTTPException
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.building_model.errors.arimax import ArimaxFitValidationError, ArimaxPydanticValidationError
from src.core.application.building_model.errors.lstm import LstmFitValidationError, LstmPydanticValidationError
from src.core.application.building_model.errors.nhits import NhitsFitValidationError, NhitsPydanticValidationError
from src.core.application.building_model.schemas.arimax import ArimaxFitRequest
from src.core.application.building_model.schemas.gru import GruFitRequest
from src.core.application.building_model.schemas.lstm import LstmFitRequest
from src.core.application.building_model.schemas.nhits import NhitsFitRequest
from src.core.application.building_model.use_cases.models import FitArimaxUC, FitGruUC, FitLstmUC, FitNhitsUC
from src.infrastructure.adapters.modeling.errors.arimax import ConstantInExogAndSpecification

fit_model_router = APIRouter(prefix="/building_model", tags=["Построение модели"])


@fit_model_router.post(
    path="/arimax/fit",
    responses={
        400: {
            "model": ArimaxFitValidationError,
            "description": "Ошибка валидации во время обучения",
        },
        422: {
            "model": ArimaxPydanticValidationError,
            "description": "Ошибка валидации запроса",
        }
    }
)
@inject_sync
def fit_arimax(
    request: ArimaxFitRequest,
    fit_arimax_uc: FromDishka[FitArimaxUC]
) -> Response:
    try:
        archive_response = fit_arimax_uc.execute(request=request)
    except ConstantInExogAndSpecification as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return Response(
        content=archive_response,
        media_type="application/octet-stream",
    )


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
    request: NhitsFitRequest,
    fit_nhits_uc: FromDishka[FitNhitsUC]
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
    request: LstmFitRequest,
    fit_lstm_uc: FromDishka[FitLstmUC]
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
    request: GruFitRequest,
    fit_gru_uc: FromDishka[FitGruUC]
) -> Response:
    archive_response = fit_gru_uc.execute(request=request)
    return Response(
        content=archive_response,
        media_type="application/octet-stream",
    )
