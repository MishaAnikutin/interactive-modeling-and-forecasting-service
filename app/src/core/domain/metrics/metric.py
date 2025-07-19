from typing import Optional

from pydantic import BaseModel, model_validator

from src.shared.utils import validate_float_param


class Metric(BaseModel):
    type: str
    value: Optional[float]

    @model_validator(mode="after")
    def validate_value(self):
        self.value = validate_float_param(self.value)
        return self
