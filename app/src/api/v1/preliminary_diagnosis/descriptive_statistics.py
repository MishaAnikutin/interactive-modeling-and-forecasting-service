from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync
from src.core.application.preliminary_diagnosis.schemas.quantiles import QuantilesResult, QuantilesParams
from src.core.application.preliminary_diagnosis.schemas.statistics import StatisticResult, \
    StatisticsResponse, StatisticsRequest, StatMetricEnum, get_russian_metric
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
    statistic: StatMetricEnum,
    request: Timeseries,
    statistics_uc: FromDishka[StatisticsUC],
) -> StatisticResult:
    rus_name = get_russian_metric(statistic)
    return statistics_uc.execute(request=request, statistic=rus_name)


@descriptive_statistics_router.post(path="/statistics")
@inject_sync
def statistics(
    request: StatisticsRequest,
    statistics_uc: FromDishka[StatisticsUC],
) -> StatisticsResponse:
    """
    Функция для обработки данных с возможностью разбиения на группы и расчета метрик.

    Параметры:
        split_option (SplitOption): Вариант разбиения данных:
            - SplitOption.NONE: обработка всего ряда без разбиения
            - SplitOption.QUARTILE: разбиение на 4 группы (квартили)
            - SplitOption.QUINTILE: разбиение на 5 групп (квинтили)
            - SplitOption.DECILE: разбиение на 10 групп (децили)
        metrics (RusStatMetricEnum | list[RusStatMetricEnum]): Какие метрики рассчитывать.
    Примечания:
            - Названия групп формируются как "номер группы + тип разбиения" (например, "1 квартиль")
            - Нумерация групп начинается с 1 (group_num+1)
    """
    return statistics_uc.execute_many(request=request)