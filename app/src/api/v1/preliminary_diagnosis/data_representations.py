from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.qq import QQResult, QQParams
from src.core.application.preliminary_diagnosis.use_cases.mean_value import MeanUC

data_representations_router = APIRouter(prefix="/data_representations", tags=["Представление данных"])


@data_representations_router.post(path="/qq")
@inject_sync
def get_qq_values(
    request: QQParams,
    kde_uc: FromDishka[MeanUC]
) -> QQResult:
    return kde_uc.execute(request=request)

