from typing import Optional

from pydantic import BaseModel, Field, model_validator

from src.core.domain import Timeseries
from src.shared.utils import validate_float_param


class CriticalValues(BaseModel):
    percent_10: Optional[float]
    percent_5: Optional[float]
    percent_1: Optional[float]

    @model_validator(mode='after')
    def validate_float(self):
        self.percent_10 = validate_float_param(self.percent_10)
        self.percent_5 = validate_float_param(self.percent_5)
        self.percent_1 = validate_float_param(self.percent_1)
        return self

class StatTestParams(BaseModel):
    ts: Timeseries

class StatTestResult(BaseModel):
    p_value: Optional[float]
    stat_value: Optional[float]
    critical_values: CriticalValues
    lags: int

    @model_validator(mode='after')
    def validate_float(self):
        self.p_value = validate_float_param(self.p_value)
        self.stat_value = validate_float_param(self.stat_value)
        return self

