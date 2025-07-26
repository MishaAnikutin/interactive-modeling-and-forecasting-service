from pydantic import BaseModel, Field, model_validator

from src.core.application.model_diagnosis.errors.jarque_bera import LowCountObservationsError
from src.core.application.model_diagnosis.schemas.common import ResidAnalysisData, StatTestResult


class JarqueBeraRequest(BaseModel):
    data: ResidAnalysisData = Field(
        title="Прогноз и исхдоные данные"
    )

    @model_validator(mode="after")
    def validate_nobs(self):
        if len(self.data.ts.values) < 2:
            raise ValueError(LowCountObservationsError().detail)
        return self


class JarqueBeraResult(StatTestResult):
    skew: float = Field(title="Значение асимметрии")
    kurtosis: float = Field(title="Значение эксцесса")
