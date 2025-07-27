from typing import Optional

from pydantic import BaseModel, Field

from src.core.application.model_diagnosis.schemas.common import ResidAnalysisData


class BreuschGodfreyRequest(BaseModel):
    data: ResidAnalysisData = Field(
        title="Прогнозы и исходные данные"
    )
    nlags: Optional[int] = Field(
        default=None,
        ge=1, le=10000,
        title="Число лагов"
    )