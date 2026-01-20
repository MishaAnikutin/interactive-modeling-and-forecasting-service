from enum import Enum
from typing import List, Dict

from pydantic import BaseModel, Field, model_validator

from src.core.domain import Timeseries


class StatisticsEnum(str, Enum):
    mean = "mean"
    median = "median"
    mode = "mode"
    variance = "variance"
    kurtosis = "kurtosis"
    skewness = "skewness"
    coefficient_of_variation = "coefficient_of_variation"


class StatisticResult(BaseModel):
    """Значение статистики"""
    value: float | dict[str, float] = Field(
        ..., title="Расчетное значение",
        examples=[3.134, {'Нижний': 123, 'Верхний': 1234}],
    )

    @model_validator(mode='after')
    def validate_value(self):
        if isinstance(self.value, float):
            self.value = round(self.value, 3)
        return self

class RusStatMetricEnum(Enum):
    N_OBS = 'Количество наблюдений'
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

    def to_rus(self) -> RusStatMetricEnum:
        """Преобразует английское название метрики в русское (по идентичным ключам)."""
        return RusStatMetricEnum[self.name]


class StatisticsRequest(BaseModel):
    metrics: List[StatMetricEnum]
    timeseries: Timeseries

class StatisticsResponse(BaseModel):
    results: Dict[RusStatMetricEnum, StatisticResult]

