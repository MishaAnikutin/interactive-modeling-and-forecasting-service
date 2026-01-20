from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync
from src.core.application.preliminary_diagnosis.schemas.quantiles import QuantilesResult, QuantilesParams
from src.core.application.preliminary_diagnosis.schemas.statistics import StatisticResult, StatisticsEnum, \
    StatisticsResponse, StatisticsRequest
from src.core.application.preliminary_diagnosis.use_cases.quantiles import QuantilesUC
from src.core.application.preliminary_diagnosis.use_cases.statistics import StatisticsUC
from src.core.domain import Timeseries

descriptive_statistics_router = APIRouter(prefix="/descriptive_statistics", tags=["Описательная статистика"])


@descriptive_statistics_router.post(path="/quantiles")
@inject_sync
def quantiles(
    request: QuantilesParams,
    quant_uc: FromDishka[QuantilesUC]
) -> QuantilesResult:
    return quant_uc.execute(request=request)


@descriptive_statistics_router.post(path="/{statistic}/")
@inject_sync
def statistic_value(
    statistic: StatisticsEnum,
    request: Timeseries,
    statistics_uc: FromDishka[StatisticsUC],
) -> StatisticResult:
    return statistics_uc.execute(request=request, statistic=statistic)


# @descriptive_statistics_router.post(path="/statistics")
# @inject_sync
# def statistics(
#     request: StatisticsRequest,
#     statistics_uc: FromDishka[StatisticsUC],
# ) -> StatisticsResponse:
#     return statistics_uc.execute_many(request=request)