from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.mean_value import MeanResult, MeanParams
from src.core.application.preliminary_diagnosis.use_cases.mean_value import MeanUC

descriptive_statistics_router = APIRouter(prefix="/descriptive_statistics", tags=["Анализ ряда на стационарность"])


@descriptive_statistics_router.post(path="/mean")
@inject_sync
def mean_value(
    request: MeanParams,
    dickey_fuller_uc: FromDishka[MeanUC]
) -> MeanResult:
    return dickey_fuller_uc.execute(request=request)

