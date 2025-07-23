from typing import Optional, Literal

from pydantic import BaseModel, Field, model_validator

from src.core.domain import Timeseries, DataFrequency


class STLParams(BaseModel):
    period: Optional[int] = Field(
        default=None, ge=2,
        description="Длина сезонного цикла; Обязательно к указанию для годовых данных."
    )
    seasonal: int = Field(
        default=7, ge=3,
        description="Нечетное число. Длина сезонного сглаживателя"
    )
    trend: Optional[int] = Field(
        default=None, ge=3,
        description="Длина трендового сглаживателя, нечетное число trend > period. "
                    "Если не указано, то вычисляется по специальной формуле."
    )
    low_pass: Optional[int] = Field(
        default=None, ge=3,
        description="Длина низкочастотного сглаживателя, нечетное число low_pass > period. "
                    "Если не задана, то берется наименьшее число большее period"
    )

    seasonal_deg: Literal["0", "1"] = Field(
        default="1",
        description="Degree of seasonal LOESS. 0 (constant) or 1 (constant and trend)."
    )
    trend_deg: Literal["0", "1"] = Field(
        default="1",
        description="Degree of trend LOESS. 0 (constant) or 1 (constant and trend)."
    )
    low_pass_deg: Literal["0", "1"] = Field(
        default="1",
        description="Degree of low pass LOESS. 0 (constant) or 1 (constant and trend)."
    )

    robust: bool = Field(
        default=False,
        description="Flag indicating whether to use a "
                    "weighted version that is robust to some forms of outliers."
    )

    seasonal_jump: int = Field(
        default=1, gt=0,
        description=(
            "Positive integer determining the linear interpolation step. "
            "If larger than 1, the LOESS is used every seasonal_jump points and "
            "linear interpolation is between fitted points. "
            "Higher values reduce estimation time.")
    )
    trend_jump: int = Field(
        default=1, gt=0,
        description=(
            "Positive integer determining the linear interpolation step. "
            "If larger than 1, the LOESS is used every seasonal_jump points and "
            "linear interpolation is between fitted points. "
            "Higher values reduce estimation time.")
    )
    low_pass_jump: int = Field(
        default=1, gt=0,
        description=(
            "Positive integer determining the linear interpolation step. "
            "If larger than 1, the LOESS is used every seasonal_jump points and "
            "linear interpolation is between fitted points. "
            "Higher values reduce estimation time.")
    )

    @model_validator(mode="after")
    def validate_seasonal_trend_pass(self):
        if self.seasonal is not None and self.seasonal % 2 == 0:
            raise ValueError("Seasonal must be odd")
        if self.trend is not None and self.trend % 2 == 0:
            raise ValueError("Trend must be odd")
        if self.low_pass is not None and self.low_pass % 2 == 0:
            raise ValueError("Low_pass must be odd")
        return self

    @model_validator(mode="after")
    def validate_low_pass(self):
        if self.low_pass is not None and self.low_pass % 2 == 0:
            raise ValueError("Low_pass must be odd")
        if self.period is not None and self.low_pass is not None and self.low_pass <= self.period:
            raise ValueError("low_pass must be an odd positive integer >= 3 where low_pass > period")
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
            raise ValueError("If data freq is year, period must be provided")
        return self

    @model_validator(mode="after")
    def validate_trend(self):
        trend = self.params.trend
        if trend is not None and trend % 2 == 0:
            raise ValueError("Trend must be odd")
        if trend and self.params.period is not None and trend <= self.params.period:
            raise ValueError("Trend must be an odd positive integer >= 3 where trend > period")
        if trend and self.params.period is None:
            if self.ts.data_frequency == DataFrequency.month and trend < 13:
                raise ValueError(
                    "Trend must be greater than or equal to 13 "
                    "if timeseries frequency is month"
                )
            elif self.ts.data_frequency == DataFrequency.quart and trend < 5:
                raise ValueError(
                    "Trend must be greater than or equal to 5 "
                    "if timeseries frequency is quarter"
                )
            elif self.ts.data_frequency == DataFrequency.year:
                raise ValueError("You should a period if the data frequency is year")
            elif self.ts.data_frequency == DataFrequency.day and trend < 365:
                raise ValueError(
                    "Trend must be greater than or equal to 365 "
                    "if timeseries frequency is day"
                )
        return self

    @model_validator(mode="after")
    def validate_low_pass(self):
        low_pass = self.params.low_pass
        if low_pass is not None and low_pass % 2 == 0:
            raise ValueError("Low_pass must be an odd positive integer >= 3 where low_pass > period")
        if self.params.period is None and low_pass is not None:
            if self.ts.data_frequency == DataFrequency.month and low_pass < 13:
                raise ValueError(
                    "Low pass must be greater than or equal to 13 "
                    "if timeseries frequency is month"
                )
            elif self.ts.data_frequency == DataFrequency.quart and low_pass < 5:
                raise ValueError(
                    "Low pass must be greater than or equal to 5 "
                    "if timeseries frequency is quarter"
                )
            elif self.ts.data_frequency == DataFrequency.year:
                raise ValueError("You should a period if the data frequency is year")
            elif self.ts.data_frequency == DataFrequency.day and low_pass < 9:
                raise ValueError(
                    "Low pass must be greater than or equal to 365 "
                    "if timeseries frequency is day"
                )
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
