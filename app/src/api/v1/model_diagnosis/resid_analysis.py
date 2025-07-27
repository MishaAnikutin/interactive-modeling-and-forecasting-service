from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync
from fastapi import APIRouter

from src.core.application.model_diagnosis.errors.arch import ArchValidationError
from src.core.application.model_diagnosis.errors.jarque_bera import JarqueBeraPydanticValidationError, JBValidationError
from src.core.application.model_diagnosis.errors.kstest import KolmogorovPydanticValidationError
from src.core.application.model_diagnosis.errors.omnibus import OmnibusPydanticValidationError
from src.core.application.model_diagnosis.schemas.arch import ArchResult, ArchRequest
from src.core.application.model_diagnosis.schemas.breusch_godfrey import BreuschGodfreyResult, BreuschGodfreyRequest
from src.core.application.model_diagnosis.schemas.common import StatTestResult
from src.core.application.model_diagnosis.schemas.jarque_bera import JarqueBeraRequest, JarqueBeraResult
from src.core.application.model_diagnosis.schemas.kstest import KolmogorovRequest
from src.core.application.model_diagnosis.schemas.omnibus import OmnibusRequest
from src.core.application.model_diagnosis.use_cases.arch import ArchUC
from src.core.application.model_diagnosis.use_cases.breusch_godfrey import AcorrBreuschGodfreyUC
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
        },
        400: {
            "model": JBValidationError,
            "description": "Ошибка в запросе"
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


@resid_analysis_router.post(
    path="/arch",
    responses={
        200: {
            "model": ArchResult,
            "description": "Успешный ответ"
        },
        400: {
            "model": ArchValidationError,
            "description": "Ошибка в запросе"
        }
    }
)
@inject_sync
def arch(
    request: ArchRequest,
    arch_uc: FromDishka[ArchUC]
) -> ArchResult:
    return arch_uc.execute(request=request)


@resid_analysis_router.post(
    path="/breusch_godfrey",
    responses={
        200: {
            "model": BreuschGodfreyResult,
            "description": "Успешный ответ"
        }

    }
)
@inject_sync
def breusch_godfrey(
    request: BreuschGodfreyRequest,
    breusch_godfrey_uc: FromDishka[AcorrBreuschGodfreyUC]
) -> BreuschGodfreyResult:
    return breusch_godfrey_uc.execute(request=request)