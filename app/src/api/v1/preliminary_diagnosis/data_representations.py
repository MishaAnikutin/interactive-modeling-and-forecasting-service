from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.pp_plot import PPplotParams, PPResult
from src.core.application.preliminary_diagnosis.schemas.qq import QQResult, QQParams
from src.core.application.preliminary_diagnosis.use_cases.pp_plot import PPplotUC
from src.core.application.preliminary_diagnosis.use_cases.qq import QQplotUC

data_representations_router = APIRouter(prefix="/data_representations", tags=["Представление данных"])


@data_representations_router.post(path="/qq")
@inject_sync
def get_qq_values(
    request: QQParams,
    qq_uc: FromDishka[QQplotUC]
) -> QQResult:
    return qq_uc.execute(request=request)

@data_representations_router.post(path="/pp")
@inject_sync
def get_pp_values(
    request: PPplotParams,
    pp_uc: FromDishka[PPplotUC]
) -> PPResult:
    return pp_uc.execute(request=request)

