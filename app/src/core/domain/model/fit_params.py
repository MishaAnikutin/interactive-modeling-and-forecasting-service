from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class FitParams(BaseModel):
    val_boundary: datetime = Field(default=datetime(2030, 11, 30))
    train_boundary: datetime = Field(default=datetime(2029, 5, 31))
    forecast_horizon: int = Field(default=12)

    @model_validator(mode='after')
    def validate_train_val(self):
        if self.val_boundary < self.train_boundary:
            raise ValueError("train_boundary должна быть раньше val_boundary")
        return self
