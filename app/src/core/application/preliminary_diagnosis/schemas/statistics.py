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
    metrics: List[RusStatMetricEnum] = Field(
        ..., title="Список статистик для расчета",
        examples=[[RusStatMetricEnum.N_OBS, RusStatMetricEnum.VAR, RusStatMetricEnum.KURTOSIS],]
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

