from typing import Optional

from pydantic import BaseModel, Field, model_validator
import numpy as np
from statsmodels.tsa.tsatools import freq_to_period

from src.core.application.generating_series.errors.naive_decomposition import ZeroOrNegativeError, MissingError, \
    LowCountObservationsError, LowCountObservationsError2
from src.core.domain import Timeseries
from enum import Enum


class ModelEnum(str, Enum):
    additive = "additive"
    multiplicative = "multiplicative"

class NaiveDecompositionParams(BaseModel):
    model: ModelEnum = Field(
        default=ModelEnum.additive,
        description="Параметр задаёт тип сезонной компоненты"
    )
    filt: Optional[list[float]] = Field(
        default=None,
        description="Параметр задаёт коэффициенты фильтра для удаления сезонной компоненты из данных. "
                    "Конкретный метод скользящего среднего, "
                    "используемый для фильтрации, определяется параметром `two_sided`. "
                    "Требование: длина списка filt (len(filt)) должна быть меньше или равна числа наблюдений"
    )
    period: Optional[int] = Field(
        default=None,
        gt=0,
        le=1000,
        description=(
            "НЕ РЕКОМЕДНУЕТСЯ его менять с пустого значения!!! "
            "Параметр указывает периодичность временного ряда "
            "(например, 1 для годовых данных, 4 для квартальных и т.д.). "
            "Ряд должен иметь как минимум 2 * period наблюдения, "
            "где period - это сколько раз в году проходит сезон ряда "
        )
    )
    two_sided: bool = Field(
        default=True,
        description=(
            "Параметр определяет метод скользящего среднего, используемый для фильтрации: "
            "- Если `True` (по умолчанию), вычисляется центрированное скользящее среднее "
            "с использованием заданных коэффициентов фильтра (`filt`). "
            "- Если `False`, коэффициенты фильтра применяются только к прошлым значениям."
        )
    )
    extrapolate_trend: Optional[int] = Field(
        default=0, ge=0, le=10000,
        description=(
            "Параметр управляет экстраполяцией тренда, полученного в результате свёртки: "
            "- Если значение > 0, тренд экстраполируется линейно методом наименьших квадратов "
            "на обоих концах (или на одном, если `two_sided=False`), "
            "используя указанное количество ближайших точек плюс одну. "
            "- Если установлено значение `'freq'`, используются ближайшие точки в количестве, равном частоте ряда. "
            "- Установка этого параметра исключает появление значений `NaN` в компонентах тренда или остатков."
        )
    )


class NaiveDecompositionRequest(BaseModel):
    ts: Timeseries = Field(title="Временной ряд для разложения")
    params: NaiveDecompositionParams = Field(title="Параметры разложения")

    @model_validator(mode="after")
    def validate_ts(self):
        x = np.array(self.ts.values)
        if not np.all(np.isfinite(x)):
            raise ValueError(MissingError().detail)
        if self.params.model == ModelEnum.multiplicative.value:
            if np.any(x <= 0):
                raise ValueError(ZeroOrNegativeError().detail)

        period = self.params.period
        if period is None:
            period = freq_to_period(self.ts.data_frequency)
        if x.shape[0] < 2 * period:
            raise ValueError(LowCountObservationsError().detail)
        if x.shape[0] < len(self.params.filt):
            raise ValueError(LowCountObservationsError2().detail)
        return self


class NaiveDecompositionResult(BaseModel):
    observed: Optional[Timeseries] = Field(
        default=Timeseries(name="Наблюдаемые значения"),
        title="Наблюдаемые значения"
    )
    seasonal: Optional[Timeseries] = Field(
        default=Timeseries(name="Сезонная компонента"),
        title="Сезонная компонента"
    )
    trend: Optional[Timeseries] = Field(
        default=Timeseries(name="Трендовая компонента"),
        title="Трендовая компонента"
    )
    resid: Optional[Timeseries] = Field(
        default=Timeseries(name="Остатки"),
        title="Остатки"
    )