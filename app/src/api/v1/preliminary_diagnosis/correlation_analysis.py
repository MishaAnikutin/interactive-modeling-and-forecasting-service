from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.corr import CorrelationAnalysisRequest, CorrelationAnalysisResponse
from src.core.application.preliminary_diagnosis.use_cases.corr import CorrelationMatrixUC

correlation_analysis_router = APIRouter(prefix="/correlation", tags=["Анализ корреляции переменных"])


@correlation_analysis_router.post(
    path="/",
    responses={
        200: {
            "model": CorrelationAnalysisResponse,
            "description": "Успешный ответ",
        }
    #     422: {
    #         "model": AcfPacfValidationError,
    #         "description": "Ошибка валидации параметров"
    #     }
    }
)
@inject_sync
def corr(
    request: CorrelationAnalysisRequest,
    corr_matrix_uc: FromDishka[CorrelationMatrixUC]
) -> CorrelationAnalysisResponse:
    return corr_matrix_uc.execute(request=request)
