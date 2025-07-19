from typing import Optional

from pydantic import BaseModel, model_validator


class Metric(BaseModel):
    type: str
    value: Optional[float]

    @model_validator(mode="after")
    def validate_value(self):
        if self.value:
            if (
                    self.value == float("inf") or
                    self.value == float("-inf") or
                    self.value == float("nan")
            ):
                self.value = None
            else:
                self.value = round(self.value, 4)
        return self
