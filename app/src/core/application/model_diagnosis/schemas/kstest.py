from pydantic import BaseModel, Field, model_validator

from src.core.application.model_diagnosis.errors.kstest import LowCountObservationsError
from src.core.application.model_diagnosis.schemas.common import ResidAnalysisData


class KolmogorovRequest(BaseModel):
    data: ResidAnalysisData = Field(
        title="Прогноз и исхдоные данные"
    )

    @model_validator(mode="after")
    def validate_nobs(self):
        if len(self.data.target.values) < 4:
            raise ValueError(LowCountObservationsError().detail)
        return self