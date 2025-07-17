from typing import Optional, Literal

from pydantic import BaseModel, Field, model_validator

from src.core.domain import Timeseries, DataFrequency


class STLParams(BaseModel):
    period: Optional[int] = Field(default=None, ge=2)
    seasonal: int = Field(default=7, ge=3)
    trend: Optional[int] = Field(default=None, ge=1)
    low_pass: Optional[int] = Field(default=None, ge=3)

    # Degree of ... LOESS. 0 (constant) or 1 (constant and trend).
    seasonal_deg: Literal["0", "1"] = Field(default="1")
    trend_deg: Literal["0", "1"] = Field(default="1")
    low_pass_deg: Literal["0", "1"] = Field(default="1")

    robust: bool = Field(default=False)

    # Positive integer determining the linear interpolation step.
    # If larger than 1, the LOESS is used every seasonal_jump points and
    # linear interpolation is between fitted points.
    # Higher values reduce estimation time.
    seasonal_jump: int = Field(default=1, gt=0)
    trend_jump: int = Field(default=1, gt=0)
    low_pass_jump: int = Field(default=1, gt=0)

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
    ts: Timeseries
    params: STLParams

    @model_validator(mode="after")
    def validate_period(self):
        if (
            self.ts.data_frequency == DataFrequency.year and
            self.params.period is None
        ):
            raise ValueError("If data freq is year, period must be provided") # если частотность годовая, то потребуем непустой период
        return self

    @model_validator(mode="after")
    def validate_trend(self):
        trend = self.params.trend
        if trend is not None and trend % 2 == 0:
            raise ValueError("Trend must be odd")
        if trend:
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
            elif self.ts.data_frequency == DataFrequency.year and trend < 1:
                raise ValueError(
                    "Trend must be greater than or equal to 1 "
                    "if timeseries frequency is year"
                )
            elif self.ts.data_frequency == DataFrequency.day and trend < 365:
                raise ValueError(
                    "Trend must be greater than or equal to 365 "
                    "if timeseries frequency is day"
                )

        return self

class STLDecompositionResult(BaseModel):
    observed: Optional[Timeseries]
    seasonal: Optional[Timeseries]
    trend: Optional[Timeseries]
    resid: Optional[Timeseries]
