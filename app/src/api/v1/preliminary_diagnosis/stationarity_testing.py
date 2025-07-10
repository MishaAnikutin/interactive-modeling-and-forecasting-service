from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.dickey_fuller import DickeyFullerParams, DickeyFullerResult
from src.core.application.preliminary_diagnosis.schemas.kpss import KpssResult, KpssParams
from src.core.application.preliminary_diagnosis.schemas.phillips_perron import PhillipsPerronParams, \
    PhillipsPerronResult
from src.core.application.preliminary_diagnosis.use_cases.dicker_fuller import DickeuFullerUC
from src.core.application.preliminary_diagnosis.use_cases.kpss import KpssUC
from src.core.application.preliminary_diagnosis.use_cases.phillips_perron import PhillipsPerronUC

stationary_testing_router = APIRouter(prefix="/stationary_testing", tags=["Анализ ряда на стационарность"])

@stationary_testing_router.post("/dickey_fuller")
@inject_sync
def dickey_fuller(
    request: DickeyFullerParams,
    dickey_fuller_uc: FromDishka[DickeuFullerUC]
) -> DickeyFullerResult:
    return dickey_fuller_uc.execute(request=request)


@stationary_testing_router.post("/kpss")
@inject_sync
def kpss(
    request: KpssParams,
    kpss_uc: FromDishka[KpssUC]
) -> KpssResult:
    return kpss_uc.execute(request=request)


@stationary_testing_router.post("/phillips_perron")
@inject_sync
def phillips_perron(
    request: PhillipsPerronParams,
    phil_perron_uc: FromDishka[PhillipsPerronUC]
) -> PhillipsPerronResult:
    return phil_perron_uc.execute(request=request)

