from pydantic import BaseModel, model_validator, Field
from src.shared.utils import validate_float_param


class Coefficient(BaseModel):
    name: str = Field(default="coefficient", title="Название коэффициента")
    value: float = Field(default=0, title="Значение коэффициента")
    p_value: float = Field(default=0.05, title="p-value коэффициента")

    @model_validator(mode="after")
    def validate_value(self):
        self.value = validate_float_param(self.value)
        self.p_value = validate_float_param(self.value)

        return self
