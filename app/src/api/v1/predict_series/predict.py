from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.predict_series.use_cases.predict_arimax import PredictArimaxUC
from src.core.application.predict_series.schemas.schemas import PredictResponse, PredictRequest


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
    request: PredictRequest,
    predict_arimax_uc: FromDishka[PredictArimaxUC]
) -> PredictResponse:
    return predict_arimax_uc.execute(request=request)
