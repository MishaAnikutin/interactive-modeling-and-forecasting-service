from datetime import datetime

from pydantic import BaseModel, Field

class FitParams(BaseModel):
    val_boundary: datetime = Field(default=datetime(2030, 11, 30))
    train_boundary: datetime = Field(default=datetime(2029, 5, 31))
    forecast_horizon: int = Field(default=12)
