from typing import Optional

from pydantic import BaseModel, model_validator, Field

from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams, CriticalValues
from src.shared.utils import validate_float_param


class RangeUnitRootParams(StatTestParams):
    @model_validator(mode="after")
    def validate_ts(self):
        if min(self.ts.values) == max(self.ts.values):
            raise ValueError("Raw is constant")
        if len(self.ts.values) < 25:
            raise ValueError("Raw should have at least 25 elements")
        return self


class RangeUnitRootResult(BaseModel):
    p_value: Optional[float] = Field(default=0.05, title="p-value теста", description="p-value теста",)
    stat_value: Optional[float] = Field(title="Значение статистики теста", description="Значение статистики теста")
    critical_values: CriticalValues = Field(
        title="Критические значения для разных уровней значимости",
        description="Критические значения для разных уровней значимости",
    )

    @model_validator(mode='after')
    def validate_float(self):
        self.p_value = validate_float_param(self.p_value)
        self.stat_value = validate_float_param(self.stat_value)
        return self