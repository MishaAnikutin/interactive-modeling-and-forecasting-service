from pydantic import Field, BaseModel, model_validator

from src.core.domain import Timeseries
from src.core.domain.stat_test import SignificanceLevel


class SpuriousRegressionRequest(BaseModel):
    dependent_variable: Timeseries = Field(..., description='Зависимая переменная')
    explanatory_variable: list[Timeseries] = Field(..., description='Обьясняющие переменные')
    r2_threshold: float = Field(default=0.2, description="Порог для R^2")
    dw_threshold: float = Field(default=0.2, description="Порог для статистики Дарбина-Уотсона")
    significance_level: SignificanceLevel = 0.05

    @model_validator(mode="after")
    def validate_value(self):
        names = [self.dependent_variable.name] + [var.name for var in self.explanatory_variable]
        n = len(names)

        if any([names[i] == names[j] for i in range(n) for j in range(n) if i != j]):
            raise ValueError("Имена переменных должны быть различными")

        return self


class SpuriousRegressionResponse(BaseModel):
    r2: float = Field(..., description="R^2")
    dw: float = Field(..., description="Статистика Дарбина-Уотсона")
    number_of_significant_coefs: int = Field(..., description="Число значимых признаков")
    is_spurious: bool = Field(..., description="Результат проверки - True если R^2 < 0.2 и DW < 0.1")
