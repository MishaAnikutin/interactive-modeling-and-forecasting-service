from enum import Enum
from typing import List, Dict, Optional

from pydantic import BaseModel, Field, model_validator

from src.core.domain import Timeseries

class StatisticResult(BaseModel):
    """Значение статистики"""
    value: Optional[float | dict[str, float]] = Field(
        ..., title="Расчетное значение",
        description="Обычно float, но для специфических случаев может быть dict или не определено",
        examples=[3.134, {'Нижний': 123, 'Верхний': 1234}, None],
    )

    @model_validator(mode='after')
    def validate_value(self):
        if isinstance(self.value, float):
            self.value = round(self.value, 3)
        return self

class RusStatMetricEnum(Enum):
    N_OBS = 'Количество наблюдений'
    VAR_COEFF = 'Коэффициент вариации'
    MEAN = 'Среднее'
    MEAN_CONF_INT = 'Доверительный интервал для среднего'
    CR_BOUND_MEAN = 'Нижняя граница неравенства Рао–Крамера для среднего (нормальное распределение)'
    STD_ERR = 'Стандартная ошибка'
    MEDIAN = 'Медиана'
    MODE = 'Мода'
    MEDIAN_WOLSH = 'Медиана Уолша'
    TRIMMED_MEAN = 'Усечённое среднее'
    STD = 'Стандартное отклонение'
    GEOM_MEAN = 'Геометрическое среднее'
    VAR = 'Дисперсия'
    VAR_CONF_INT = 'Доверительный интервал для дисперсии'
    CR_BOUND_VAR = 'Нижняя граница неравенства Рао–Крамера для дисперсии (нормальное распределение)'
    KURTOSIS = 'Эксцесс'
    SKEW = 'Асимметрия'
    MIN = 'Минимум'
    MAX = 'Максимум'
    RANGE = 'Размах'
    SUM = 'Сумма'
    Q25 = '25% квантиль'
    Q75 = '75% квантиль'
    LAST_Z = 'Последний z-показатель'
    ENTROPY = 'Энтропия'

class StatMetricEnum(Enum):
    N_OBS = 'Number of observations'
    VAR_COEFF = 'Coefficient of variation'
    MEAN = 'Mean'
    MEAN_CONF_INT = 'Mean confidence interval'
    CR_BOUND_MEAN = 'Rao-Cramer inequality lower bound mean for normal distribution'
    STD_ERR = 'Standard error'
    MEDIAN = 'Median'
    MODE = 'Mode'
    MEDIAN_WOLSH = 'Median Wolsh'
    TRIMMED_MEAN = 'Trimmed Mean'
    STD = 'Standard Deviation'
    GEOM_MEAN = 'Geometric mean'
    VAR = 'Variance'
    VAR_CONF_INT = 'Variance confident interval'
    CR_BOUND_VAR = 'Rao-Cramer inequality lower bound variance for normal distribution'
    KURTOSIS = 'Excess kurtosis'
    SKEW = 'Skewness'
    MIN = 'Minimum'
    MAX = 'Maximum'
    RANGE = 'Range'
    SUM = 'Sum'
    Q25 = '25% quantile'
    Q75 = '75% quantile'
    LAST_Z = 'Last z-score'
    ENTROPY = 'Entropy'

STAT_METRIC_TO_RUS = {
    StatMetricEnum.N_OBS: RusStatMetricEnum.N_OBS,
    StatMetricEnum.VAR_COEFF: RusStatMetricEnum.VAR_COEFF,
    StatMetricEnum.MEAN: RusStatMetricEnum.MEAN,
    StatMetricEnum.MEAN_CONF_INT: RusStatMetricEnum.MEAN_CONF_INT,
    StatMetricEnum.CR_BOUND_MEAN: RusStatMetricEnum.CR_BOUND_MEAN,
    StatMetricEnum.STD_ERR: RusStatMetricEnum.STD_ERR,
    StatMetricEnum.MEDIAN: RusStatMetricEnum.MEDIAN,
    StatMetricEnum.MODE: RusStatMetricEnum.MODE,
    StatMetricEnum.MEDIAN_WOLSH: RusStatMetricEnum.MEDIAN_WOLSH,
    StatMetricEnum.TRIMMED_MEAN: RusStatMetricEnum.TRIMMED_MEAN,
    StatMetricEnum.STD: RusStatMetricEnum.STD,
    StatMetricEnum.GEOM_MEAN: RusStatMetricEnum.GEOM_MEAN,
    StatMetricEnum.VAR: RusStatMetricEnum.VAR,
    StatMetricEnum.VAR_CONF_INT: RusStatMetricEnum.VAR_CONF_INT,
    StatMetricEnum.CR_BOUND_VAR: RusStatMetricEnum.CR_BOUND_VAR,
    StatMetricEnum.KURTOSIS: RusStatMetricEnum.KURTOSIS,
    StatMetricEnum.SKEW: RusStatMetricEnum.SKEW,
    StatMetricEnum.MIN: RusStatMetricEnum.MIN,
    StatMetricEnum.MAX: RusStatMetricEnum.MAX,
    StatMetricEnum.RANGE: RusStatMetricEnum.RANGE,
    StatMetricEnum.SUM: RusStatMetricEnum.SUM,
    StatMetricEnum.Q25: RusStatMetricEnum.Q25,
    StatMetricEnum.Q75: RusStatMetricEnum.Q75,
    StatMetricEnum.LAST_Z: RusStatMetricEnum.LAST_Z,
    StatMetricEnum.ENTROPY: RusStatMetricEnum.ENTROPY,
}

def get_russian_metric(metric: StatMetricEnum) -> RusStatMetricEnum:
    """Конвертирует английскую метрику в русскую."""
    return STAT_METRIC_TO_RUS.get(metric, f"Неизвестная метрика {metric}")

class SplitOption(Enum):
    """
    Вариант разбиения данных:
        - SplitOption.NONE: обработка всего ряда без разбиения
        - SplitOption.QUARTILE: разбиение на 4 группы (квартили)
        - SplitOption.QUINTILE: разбиение на 5 групп (квинтили)
        - SplitOption.DECILE: разбиение на 10 групп (децили)
    """
    NONE = None
    QUARTILE = 'квартили'
    QUINTILE = 'квинтили'
    DECILE = 'децили'

class StatisticsRequest(BaseModel):
    metrics: List[StatMetricEnum] = Field(
        ..., title="Список статистик для расчета",
        examples=[[StatMetricEnum.N_OBS, StatMetricEnum.VAR, StatMetricEnum.KURTOSIS],]
    )
    split_option: SplitOption = Field(
        ..., title="Вариант разбиения данных",
        description=(
            """
            Вариант разбиения данных:
                - SplitOption.NONE: обработка всего ряда без разбиения:
                - SplitOption.QUARTILE: разбиение на 4 группы (квартили),
                - SplitOption.QUINTILE: разбиение на 5 групп (квинтили),
                - SplitOption.DECILE: разбиение на 10 групп (децили),
            """
        ),
        examples=[SplitOption.NONE, SplitOption.QUINTILE]
    )
    timeseries: Timeseries = Field(..., title="Временной ряд")

class StatisticsResponse(BaseModel):
    results: Dict[str, Dict[RusStatMetricEnum, StatisticResult]] = Field(
        ..., title="Результаты расчетов",
        examples=[
            {
                "квартиль 1": {"Минимум": {"value": 1.2}},
                "квартиль 2": {"Минимум": {"value": 2.2}},
                "квартиль 3": {"Минимум": {"value": 3.2}},
                "квартиль 4": {"Минимум": {"value": 4.2}},
            }
        ]
    )

