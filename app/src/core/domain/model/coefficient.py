from pydantic import BaseModel, model_validator, Field
from src.shared.utils import validate_float_param


class Coefficient(BaseModel):
    name: str = Field(default="coefficient", title="Название коэффициента")
    value: float
    p_value: float

    @model_validator(mode="after")
    def validate_value(self):
        self.value = validate_float_param(self.value)
        self.p_value = validate_float_param(self.value)

        return self
