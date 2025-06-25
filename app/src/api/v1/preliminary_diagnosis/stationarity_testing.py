from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.stats_test import StatTestRequest, StatTestResult

stationary_testing_router = APIRouter(prefix="/stationary_testing", tags=["Анализ ряда на стационарность"])


@stationary_testing_router.get("/dickey_fuller")
@inject_sync
def dickey_fuller(
    request: StatTestRequest,
    stationary_testing_uc: FromDishka[StationaryTestingUC]
) -> StatTestResult:
    return stationary_testing_uc.dickey_fuller(request)
