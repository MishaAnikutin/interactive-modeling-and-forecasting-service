from fastapi import APIRouter, File, UploadFile
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.predict_series.schemas.schemas import PredictRequest
from src.core.application.predict_series.use_cases.predict_gru import PredictGruUC_V2
from src.core.application.predict_series.use_cases.predict_lstm import PredictLstmUC_V2
from src.core.application.predict_series.use_cases.predict_nhits import PredictNhitsUC_V2
from src.core.domain import ForecastResult_V2

model_predict_router = APIRouter(prefix="/model_predicting", tags=["Прогнозы моделей"])

@model_predict_router.post(path="/gru/predict")
@inject_sync
def predict_gru(
    predict_gru_uc: FromDishka[PredictGruUC_V2],
    request: PredictRequest,
    model_file: UploadFile = File(..., description="Файл модели в формате .pickle"),
) -> ForecastResult_V2:
    return predict_gru_uc.execute(request=request, model_bytes=model_file.file.read())


@model_predict_router.post(path="/nhits/predict")
@inject_sync
def predict_nhits(
    predict_nhits_uc: FromDishka[PredictNhitsUC_V2],
    request: PredictRequest,
    model_file: UploadFile = File(..., description="Файл модели в формате .pickle"),
) -> ForecastResult_V2:
    return predict_nhits_uc.execute(request=request, model_bytes=model_file.file.read())


@model_predict_router.post(path="/lstm/predict")
@inject_sync
def predict_lstm(
    predict_lstm_uc: FromDishka[PredictLstmUC_V2],
    request: PredictRequest,
    model_file: UploadFile = File(..., description="Файл модели в формате .pickle"),
) -> ForecastResult_V2:
    return predict_lstm_uc.execute(request=request, model_bytes=model_file.file.read())
