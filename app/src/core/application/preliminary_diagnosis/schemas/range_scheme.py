from pydantic import BaseModel, model_validator

from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams, CriticalValues


class RangeUnitRootParams(StatTestParams):
    @model_validator(mode="after")
    def validate_ts(self):
        if min(self.ts.values) == max(self.ts.values):
            raise ValueError("Raw is constant")
        if len(self.ts.values) < 25:
            raise ValueError("Raw should have at least 25 elements")
        return self


class RangeUnitRootResult(BaseModel):
    p_value: float
    stat_value: float
    critical_values: CriticalValues
