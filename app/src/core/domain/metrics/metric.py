from typing import Optional

from pydantic import BaseModel, model_validator, Field

from src.shared.utils import validate_float_param


class Metric(BaseModel):
    type: str = Field(default="MAPE", description="Тип метрики")
    value: Optional[float] = Field(default=0, description="Значение метрики")

    @model_validator(mode="after")
    def validate_value(self):
        self.value = validate_float_param(self.value)
        return self
