from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.stats_test import StatTestRequest, StatTestResult
from src.core.application.preliminary_diagnosis.use_cases.stats_test import StationarityUC

stationary_testing_router = APIRouter(prefix="/stationary_testing", tags=["Анализ ряда на стационарность"])


@stationary_testing_router.get("/")
@inject_sync
def stationary_testing(
    request: StatTestRequest,
    stationary_testing_uc: FromDishka[StationarityUC]
) -> StatTestResult:
    return stationary_testing_uc.execute(request=request)
