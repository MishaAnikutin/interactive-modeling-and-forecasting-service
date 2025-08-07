from fastapi import APIRouter, UploadFile, File, Body
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.predict_series.use_cases.predict_arimax import PredictArimaxUC
from src.core.application.predict_series.use_cases.predict_gru import PredictGruUC
from src.core.application.predict_series.schemas.schemas import PredictResponse, PredictArimaxRequest, PredictGruRequest


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
def predict_arimax(
    predict_arimax_uc: FromDishka[PredictArimaxUC],
    request: PredictArimaxRequest = Body(...),
    model_file: UploadFile = File(..., description="Файл модели в формате .pickle"),
) -> PredictResponse:
    return predict_arimax_uc.execute(request=request, model_bytes=model_file.file.read())


# FIXME: пока закоммитил, чтобы у бэкендеров не было путаницы пока у нас схемы разыне
# @model_predict_router.post(
#     path="/gru/predict",
#     responses={
#         200: {
#             "model": PredictResponse,
#             "description": "Успешный ответ"
#         },
#     }
# )
@inject_sync
def fit_gru(
    request: PredictGruRequest,
    predict_gru_uc: FromDishka[PredictGruUC]
) -> PredictResponse:
    return predict_gru_uc.execute(request=request)
