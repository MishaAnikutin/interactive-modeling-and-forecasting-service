from typing import Optional, Literal

from pydantic import BaseModel, Field, model_validator

from src.core.application.generating_series.errors.stl_decomposition import NonePeriodError, InvalidTrendError, \
    InvalidLowPassError
from src.core.domain import Timeseries, DataFrequency


class STLParams(BaseModel):
    period: Optional[int] = Field(
        default=None, ge=2,
        description="Длина сезонного цикла; "
                    "Требование: Если ряд имеет годовую частотность, то period должен быть не пустым"
    )
    seasonal: int = Field(
        default=7, ge=3,
        description="Нечетное число. Длина сезонного сглаживателя. Нечетное число."
    )
    trend: Optional[int] = Field(
        default=None, ge=3,
        description="Длина трендового сглаживателя, нечетное число. "
                    "Trend должен быть целым число >= 3, которое больше period, если period указан."
                    "Если period не указан, то: "
                    "1) при дневной частотности period >= 365 "
                    "2) при месячной частотности period >= 13 "
                    "3) при квартальной частотности period >= 5 "
                    "4) при годовой частотности надо выбрать period согласно правилу выше"
    )
    low_pass: Optional[int] = Field(
        default=None, ge=3,
        description="Длина низкочастотного сглаживателя, нечетное число low_pass > period. "
                    "Low pass должен быть целым число >= 3, которое больше period, если period указан. "
                    "Если period не указан, то: "
                    "1) при дневной частотности low_pass >= 9 "
                    "2) при месячной частотности low_pass >= 13 "
                    "3) при квартальной частотности low_pass >= 5 "
                    "4) при годовой частотности надо выбрать period согласно правилу выше"
    )

    seasonal_deg: Literal["0", "1"] = Field(
        default="1",
        description="Параметр определяет степень сезонного LOESS. 0 (constant) or 1 (constant and trend)."
    )
    trend_deg: Literal["0", "1"] = Field(
        default="1",
        description="Параметр определяет степень сезонного LOESS. 0 (constant) or 1 (constant and trend)."
    )
    low_pass_deg: Literal["0", "1"] = Field(
        default="1",
        description="Параметр определяет степень сезонного LOESS. 0 (constant) or 1 (constant and trend)."
    )

    robust: bool = Field(
        default=False,
        description="Параметр указывает, использовать ли взвешенную версию метода, "
                    "устойчивую к некоторым видам выбросов."
    )

    seasonal_jump: int = Field(
        default=1, gt=0,
        description=(
            "Параметр — положительное целое число, определяющее шаг линейной интерполяции: "
            "- Если значение больше 1, метод LOESS применяется с интервалом в `seasonal_jump` точек, "
            "а между рассчитанными точками используется линейная интерполяция. "
            "- Более высокие значения уменьшают время вычислений."
        )
    )
    trend_jump: int = Field(
        default=1, gt=0, le=10000,
        description=(
            "Параметр — положительное целое число, определяющее шаг линейной интерполяции: "
            "- Если значение больше 1, метод LOESS применяется с интервалом в `seasonal_jump` точек, "
            "а между рассчитанными точками используется линейная интерполяция. "
            "- Более высокие значения уменьшают время вычислений."
        )
    )
    low_pass_jump: int = Field(
        default=1, gt=0, le=10000,
        description=(
            "Параметр — положительное целое число, определяющее шаг линейной интерполяции: "
            "- Если значение больше 1, метод LOESS применяется с интервалом в `seasonal_jump` точек, "
            "а между рассчитанными точками используется линейная интерполяция. "
            "- Более высокие значения уменьшают время вычислений."
        )
    )

    @model_validator(mode="after")
    def validate_seasonal_trend_pass(self):
        if self.seasonal is not None and self.seasonal % 2 == 0:
            raise ValueError("Seasonal должен быть нечетным числом")
        if self.trend is not None and self.trend % 2 == 0:
            raise ValueError("Trend должен быть нечетным числом")
        if self.low_pass is not None and self.low_pass % 2 == 0:
            raise ValueError("Low_pass должен быть нечетным числом")
        return self

    @model_validator(mode="after")
    def validate_low_pass(self):
        if self.low_pass is not None and self.low_pass % 2 == 0:
            raise ValueError(InvalidLowPassError().detail)
        if self.period is not None and self.low_pass is not None and self.low_pass <= self.period:
            raise ValueError(InvalidLowPassError().detail)
        return self

class STLDecompositionRequest(BaseModel):
    ts: Timeseries = Field(title="Временной ряд для разложения")
    params: STLParams = Field(title="Параметры разложения")

    @model_validator(mode="after")
    def validate_period(self):
        if (
            self.ts.data_frequency == DataFrequency.year and
            self.params.period is None
        ):
            raise ValueError(NonePeriodError().detail)
        return self

    @model_validator(mode="after")
    def validate_trend(self):
        trend = self.params.trend
        if trend is not None and trend % 2 == 0:
            raise ValueError(InvalidTrendError().detail)
        if trend and self.params.period is not None and trend <= self.params.period:
            raise ValueError(InvalidTrendError().detail)
        if trend and self.params.period is None:
            if self.ts.data_frequency == DataFrequency.month and trend < 13:
                raise ValueError(InvalidTrendError().detail)
            elif self.ts.data_frequency == DataFrequency.quart and trend < 5:
                raise ValueError(InvalidTrendError().detail)
            elif self.ts.data_frequency == DataFrequency.year:
                raise ValueError(NonePeriodError().detail)
            elif self.ts.data_frequency == DataFrequency.day and trend < 365:
                raise ValueError(InvalidTrendError().detail)
        return self

    @model_validator(mode="after")
    def validate_low_pass(self):
        low_pass = self.params.low_pass
        if low_pass is not None and low_pass % 2 == 0:
            raise ValueError(InvalidTrendError().detail)
        if self.params.period is None and low_pass is not None:
            if self.ts.data_frequency == DataFrequency.month and low_pass < 13:
                raise ValueError(InvalidLowPassError().detail)
            elif self.ts.data_frequency == DataFrequency.quart and low_pass < 5:
                raise ValueError(InvalidLowPassError().detail)
            elif self.ts.data_frequency == DataFrequency.year:
                raise ValueError(NonePeriodError().detail)
            elif self.ts.data_frequency == DataFrequency.day and low_pass < 9:
                raise ValueError(InvalidLowPassError().detail)
        return self

class STLDecompositionResult(BaseModel):
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
