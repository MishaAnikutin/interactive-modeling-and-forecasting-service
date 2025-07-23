from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.errors.df_gls import DfGlsPydanticValidationError, \
    DfGlsExecuteValidationError
from src.core.application.preliminary_diagnosis.errors.dickey_fuller import DickeyFullerPydanticValidationError
from src.core.application.preliminary_diagnosis.schemas.common import StatTestResult
from src.core.application.preliminary_diagnosis.schemas.df_gls import DfGlsParams
from src.core.application.preliminary_diagnosis.schemas.dickey_fuller import DickeyFullerParams, DickeyFullerResult
from src.core.application.preliminary_diagnosis.schemas.kpss import KpssParams
from src.core.application.preliminary_diagnosis.schemas.phillips_perron import PhillipsPerronParams
from src.core.application.preliminary_diagnosis.schemas.range_scheme import RangeUnitRootResult, RangeUnitRootParams
from src.core.application.preliminary_diagnosis.schemas.zivot_andrews import ZivotAndrewsParams
from src.core.application.preliminary_diagnosis.use_cases.df_gls import DfGlsUC
from src.core.application.preliminary_diagnosis.use_cases.dicker_fuller import DickeuFullerUC
from src.core.application.preliminary_diagnosis.use_cases.kpss import KpssUC
from src.core.application.preliminary_diagnosis.use_cases.phillips_perron import PhillipsPerronUC
from src.core.application.preliminary_diagnosis.use_cases.range_uc import RangeUnitRootUC
from src.core.application.preliminary_diagnosis.use_cases.zivot_andrews import ZivotAndrewsUC

stationary_testing_router = APIRouter(prefix="/stationary_testing", tags=["Анализ ряда на стационарность"])

@stationary_testing_router.post(
    path="/dickey_fuller",
    responses={
        200: {
            "model": DickeyFullerResult,
            "description": "Успешный ответ",
        },
        422: {
            "model": DickeyFullerPydanticValidationError,
            "description": "Ошибка валидации параметров"
        }
    }
)
@inject_sync
def dickey_fuller(
    request: DickeyFullerParams,
    dickey_fuller_uc: FromDishka[DickeuFullerUC]
) -> DickeyFullerResult:
    return dickey_fuller_uc.execute(request=request)


@stationary_testing_router.post(
    path="/kpss",
    responses={
        200: {
            "model": StatTestResult,
            "description": "Успешный ответ",
        },
        422: {
            "model": DickeyFullerPydanticValidationError,
            "description": "Ошибка валидации параметров"
        }
    }
)
@inject_sync
def kpss(
    request: KpssParams,
    kpss_uc: FromDishka[KpssUC]
) -> StatTestResult:
    return kpss_uc.execute(request=request)


@stationary_testing_router.post("/phillips_perron")
@inject_sync
def phillips_perron(
    request: PhillipsPerronParams,
    phil_perron_uc: FromDishka[PhillipsPerronUC]
) -> StatTestResult:
    return phil_perron_uc.execute(request=request)

@stationary_testing_router.post(
    path="/df_gls",
    responses={
        200: {
            "model": StatTestResult,
            "description": "Успешный ответ",
        },
        422: {
            "model": DfGlsPydanticValidationError,
            "description": "Ошибка валидации параметров"
        },
        400: {
            "model": DfGlsExecuteValidationError,
            "description": "Ошибка выполнения теста"
        }
    }
)
@inject_sync
def df_gls(
    request: DfGlsParams,
    df_gls_uc: FromDishka[DfGlsUC]
) -> StatTestResult:
    return df_gls_uc.execute(request=request)


@stationary_testing_router.post("/zivot_andrews")
@inject_sync
def zivot_andrews(
    request: ZivotAndrewsParams,
    zivot_andrews_uc: FromDishka[ZivotAndrewsUC]
) -> StatTestResult:
    return zivot_andrews_uc.execute(request=request)

@stationary_testing_router.post("/range_unit_root")
@inject_sync
def range_unit_root(
    request: RangeUnitRootParams,
    range_unit_root_uc: FromDishka[RangeUnitRootUC]
) -> RangeUnitRootResult:
    return range_unit_root_uc.execute(request=request)
