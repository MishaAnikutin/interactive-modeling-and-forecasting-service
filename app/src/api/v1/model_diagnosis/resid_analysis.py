from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync
from fastapi import APIRouter

from src.core.application.model_diagnosis.errors.omnibus import OmnibusPydanticValidationError
from src.core.application.model_diagnosis.schemas.common import StatTestResult
from src.core.application.model_diagnosis.schemas.omnibus import OmnibusRequest
from src.core.application.model_diagnosis.use_cases.omnibus import OmnibusUC

resid_analysis_router = APIRouter(
    prefix="/resid_analysis",
    tags=["Анализ остатков"]
)


@resid_analysis_router.post(
    path="/omnibus",
    responses={
        200: {
            "model": StatTestResult,
            "description": "Успешный ответ"
        },
        422: {
            "model": OmnibusPydanticValidationError,
            "description": "Ошибка валидации параметров"
        }
    }
)
@inject_sync
def resid_analysis(
    request: OmnibusRequest,
    omnibus_uc: FromDishka[OmnibusUC]
) -> StatTestResult:
    return omnibus_uc.execute(request=request)