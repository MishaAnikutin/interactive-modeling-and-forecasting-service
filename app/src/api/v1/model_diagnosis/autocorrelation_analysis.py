from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync


from src.core.application.model_diagnosis.errors.common import ResidAnalysisValidationError
from src.core.application.model_diagnosis.schemas.common import DiagnosticsResult
from src.core.application.model_diagnosis.errors.ljung_box import LBPydanticValidationError
from src.core.application.model_diagnosis.schemas.breusch_godfrey import BreuschGodfreyRequest
from src.core.application.model_diagnosis.schemas.ljung_box import LjungBoxResult, LjungBoxRequest
from src.core.application.model_diagnosis.use_cases.breusch_godfrey import AcorrBreuschGodfreyUC
from src.core.application.model_diagnosis.use_cases.ljung_box import LjungBoxUC


autocorrelation_router = APIRouter(
    prefix="/autocorrelation_analysis",
    tags=["Анализ автокорреляции"]
)


@autocorrelation_router.post(
    path='/ljung_box',
    responses={
        200: {
            "model": LjungBoxResult,
            "description": "Успешный ответ"
        },
        400: {
            "model": ResidAnalysisValidationError,
            "description": "Ошибка в запросе"
        },
        422: {
            "model": LBPydanticValidationError,
            "description": "Ошибка в параметрах"
        }
    }
)
@inject_sync
def ljung_box(
    request: LjungBoxRequest,
    ljung_box_uc: FromDishka[LjungBoxUC]
) -> LjungBoxResult:
    return ljung_box_uc.execute(request=request)


@autocorrelation_router.post(
    path="/breusch_godfrey",
    responses={
        200: {
            "model": DiagnosticsResult,
            "description": "Успешный ответ"
        },
        400: {
            "model": ResidAnalysisValidationError,
            "description": "Ошибка в запросе"
        }
    }
)
@inject_sync
def breusch_godfrey(
    request: BreuschGodfreyRequest,
    breusch_godfrey_uc: FromDishka[AcorrBreuschGodfreyUC]
) -> DiagnosticsResult:
    return breusch_godfrey_uc.execute(request=request)
