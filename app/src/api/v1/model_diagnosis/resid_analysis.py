from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync
from fastapi import APIRouter

from src.core.application.model_diagnosis.errors.jarque_bera import JarqueBeraPydanticValidationError
from src.core.application.model_diagnosis.errors.kstest import KolmogorovPydanticValidationError
from src.core.application.model_diagnosis.errors.omnibus import OmnibusPydanticValidationError
from src.core.application.model_diagnosis.schemas.common import StatTestResult
from src.core.application.model_diagnosis.schemas.jarque_bera import JarqueBeraRequest, JarqueBeraResult
from src.core.application.model_diagnosis.schemas.kstest import KolmogorovRequest
from src.core.application.model_diagnosis.schemas.omnibus import OmnibusRequest
from src.core.application.model_diagnosis.use_cases.jarque_bera import JarqueBeraUC
from src.core.application.model_diagnosis.use_cases.kstest import KolmogorovUC
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


@resid_analysis_router.post(
    path="/jarque_bera",
    responses={
        200: {
            "model": JarqueBeraResult,
            "description": "Успешный ответ"
        },
        422: {
            "model": JarqueBeraPydanticValidationError,
            "description": "Ошибка валидации параметров"
        }
    }
)
@inject_sync
def jarque_bera(
    request: JarqueBeraRequest,
    jarque_bera_uc: FromDishka[JarqueBeraUC]
) -> StatTestResult:
    return jarque_bera_uc.execute(request=request)


@resid_analysis_router.post(
    path="/kstest_normal",
    responses={
        200: {
            "model": StatTestResult,
            "description": "Успешный ответ"
        },
        422: {
            "model": KolmogorovPydanticValidationError,
            "description": "Ошибка валидации параметров"
        }
    }
)
@inject_sync
def kstest_normal(
    request: KolmogorovRequest,
    kolmogorov_uc: FromDishka[KolmogorovUC]
) -> StatTestResult:
    return kolmogorov_uc.execute(request=request)