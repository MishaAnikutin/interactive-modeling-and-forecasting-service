from pydantic import BaseModel, model_validator, Field

from src.core.application.model_diagnosis.errors.omnibus import LowCountObservationsError
from src.core.application.model_diagnosis.schemas.common import ResidAnalysisData


class OmnibusRequest(BaseModel):
    data: ResidAnalysisData = Field(
        title="Прогноз и исхдоные данные"
    )

    @model_validator(mode="after")
    def validate_nobs(self):
        if len(self.data.ts.values) < 8:
            raise ValueError(LowCountObservationsError().detail)
        return self

