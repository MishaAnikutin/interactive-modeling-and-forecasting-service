from fastapi import APIRouter, UploadFile, File
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.predict_series.use_cases.predict_arimax import PredictArimaxUC
from src.core.application.predict_series.use_cases.predict_gru import PredictGruUC
from src.core.application.predict_series.schemas.schemas import PredictRequest
from src.core.application.predict_series.use_cases.predict_lstm import PredictLstmUC
from src.core.application.predict_series.use_cases.predict_nhits import PredictNhitsUC
from src.core.domain import ForecastResult

model_predict_router = APIRouter(prefix="/model_predicting", tags=["Прогнозы моделей"])


@model_predict_router.post(
    path="/arimax/predict",
    responses={
        200: {
            "model": ForecastResult,
            "description": "Успешный ответ"
        },
    }
)
@inject_sync
def predict_arimax(
    predict_arimax_uc: FromDishka[PredictArimaxUC],
    request: PredictRequest,
    model_file: UploadFile = File(..., description="Файл модели в формате .pickle"),
) -> ForecastResult:
    return predict_arimax_uc.execute(request=request, model_bytes=model_file.file.read())


@model_predict_router.post(
    path="/gru/predict",
    responses={
        200: {
            "model": ForecastResult,
            "description": "Успешный ответ"
        },
    }
)
@inject_sync
def predict_gru(
    predict_gru_uc: FromDishka[PredictGruUC],
    request: PredictRequest,
    model_file: UploadFile = File(..., description="Файл модели в формате .pickle"),
) -> ForecastResult:
    return predict_gru_uc.execute(request=request, model_bytes=model_file.file.read())


@model_predict_router.post(
    path="/nhits/predict",
    responses={
        200: {
            "model": ForecastResult,
            "description": "Успешный ответ"
        },
    }
)
@inject_sync
def predict_nhits(
    predict_nhits_uc: FromDishka[PredictNhitsUC],
    request: PredictRequest,
    model_file: UploadFile = File(..., description="Файл модели в формате .pickle"),
) -> ForecastResult:
    return predict_nhits_uc.execute(request=request, model_bytes=model_file.file.read())


@model_predict_router.post(
    path="/lstm/predict",
    responses={
        200: {
            "model": ForecastResult,
            "description": "Успешный ответ"
        },
    }
)
@inject_sync
def predict_lstm(
    predict_lstm_uc: FromDishka[PredictLstmUC],
    request: PredictRequest,
    model_file: UploadFile = File(..., description="Файл модели в формате .pickle"),
) -> ForecastResult:
    return predict_lstm_uc.execute(request=request, model_bytes=model_file.file.read())