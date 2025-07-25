from datetime import date

from pydantic import BaseModel, Field, model_validator


class FitParams(BaseModel):
    val_boundary: date = Field(default=date(2030, 11, 30), title="Граница валидационной выборки")
    train_boundary: date = Field(default=date(2029, 5, 31), title="Граница обучающей выборки")
    forecast_horizon: int = Field(default=12, title="Горизонт прогноза", ge=0, le=1000)

    @model_validator(mode='after')
    def validate_train_val(self):
        if self.val_boundary < self.train_boundary:
            raise ValueError("train_boundary должна быть раньше val_boundary")
        return self
