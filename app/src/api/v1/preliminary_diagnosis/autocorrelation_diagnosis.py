from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.errors.acf_and_pacf import AcfPacfValidationError
from src.core.application.preliminary_diagnosis.schemas.acf_and_pacf import AcfAndPacfRequest, AcfPacfResult
from src.core.application.preliminary_diagnosis.use_cases.acf_and_pacf import AcfAndPacfUC

autocorrelation_diagnosis_router = APIRouter(prefix="/autocorrelation_diagnosis", tags=["Анализ автокорреляции"])


@autocorrelation_diagnosis_router.post(
    path="/acf_and_pacf",
    responses={
        200: {
            "model": AcfPacfResult,
            "description": "Успешный ответ",
        },
        422: {
            "model": AcfPacfValidationError,
            "description": "Ошибка валидации параметров"
        }
    }
)
@inject_sync
def acf(
    request: AcfAndPacfRequest,
    acf_pacf_uc: FromDishka[AcfAndPacfUC]
) -> AcfPacfResult:
    return acf_pacf_uc.execute(request=request)
