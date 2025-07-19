from typing import Optional

from pydantic import BaseModel, Field, model_validator
import numpy as np
from statsmodels.tsa.tsatools import freq_to_period

from src.core.domain import Timeseries
from enum import Enum


class ModelEnum(str, Enum):
    additive = "additive"
    multiplicative = "multiplicative"

class NaiveDecompositionParams(BaseModel):
    model: ModelEnum = Field(
        default=ModelEnum.additive,
        description="Type of seasonal component. Abbreviations are accepted."
    )
    filt: Optional[list[float]] = Field(
        default=None,
        description="The filter coefficients for filtering out the seasonal component. "
                    "The concrete moving average method used in filtering is determined by two_sided."
    )
    period: Optional[int] = Field(
        default=None,
        gt=0,
        description="Period of the series (e.g., 1 for annual, 4 for quarterly, etc). "
                    "Must be used if x is not a pandas object or "
                    "if the index of x does not have a frequency. "
                    "Overrides default periodicity of x if x is a pandas object with a timeseries index."
    )
    two_sided: bool = Field(
        default=True,
        description="The moving average method used in filtering. "
                    "If True (default), a centered moving average is computed using the filt. "
                    "If False, the filter coefficients are for past values only."
    )
    extrapolate_trend: Optional[int] = Field(
        default=0, ge=0,
        description="If set to > 0, the trend resulting from the convolution is linear "
                    "least-squares extrapolated on both ends "
                    "(or the single one if two_sided is False) "
                    "considering this many (+1) closest points. "
                    "If set to ‘freq’, use freq closest points. "
                    "Setting this parameter results in no NaN values in trend or "
                    "resid components."
    )


class NaiveDecompositionRequest(BaseModel):
    ts: Timeseries
    params: NaiveDecompositionParams

    @model_validator(mode="after")
    def validate_ts(self):
        x = np.array(self.ts.values)
        if not np.all(np.isfinite(x)):
            raise ValueError("This function does not handle missing values")
        if self.params.model == ModelEnum.multiplicative.value:
            if np.any(x <= 0):
                raise ValueError(
                    "Multiplicative seasonality is not appropriate "
                    "for zero and negative values"
                )

        period = self.params.period
        if period is None:
            period = freq_to_period(self.ts.data_frequency)
        if x.shape[0] < 2 * period:
            raise ValueError(
                f"x must have 2 complete cycles requires {2 * period} "
                f"observations. x only has {x.shape[0]} observation(s)"
            )
        return self


class NaiveDecompositionResult(BaseModel):
    observed: Optional[Timeseries]
    seasonal: Optional[Timeseries]
    trend: Optional[Timeseries]
    resid: Optional[Timeseries]