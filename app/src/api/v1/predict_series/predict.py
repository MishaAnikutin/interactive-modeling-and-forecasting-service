from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.predict_series.use_cases.predict_arimax import PredictArimaxUC
from src.core.application.predict_series.schemas.schemas import PredictResponse, PredictArimaxRequest, PredictGruRequest
from src.core.application.predict_series.use_cases.predict_gru import PredictGruUC

model_predict_router = APIRouter(prefix="/model_predicting", tags=["Прогнозы моделей"])


@model_predict_router.post(
    path="/arimax/predict",
    responses={
        200: {
            "model": PredictResponse,
            "description": "Успешный ответ"
        },
    }
)
@inject_sync
def fit_arimax(
    request: PredictArimaxRequest,
    predict_arimax_uc: FromDishka[PredictArimaxUC]
) -> PredictResponse:
    return predict_arimax_uc.execute(request=request)

@model_predict_router.post(
    path="/gru/predict",
    responses={
        200: {
            "model": PredictResponse,
            "description": "Успешный ответ"
        },
    }
)
@inject_sync
def fit_gru(
    request: PredictGruRequest,
    predict_gru_uc: FromDishka[PredictGruUC]
) -> PredictResponse:
    return predict_gru_uc.execute(request=request)
