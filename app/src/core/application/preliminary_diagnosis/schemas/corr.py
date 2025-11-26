from typing import List

from pydantic import BaseModel, Field, model_validator

from src.core.domain import Timeseries
from src.core.domain.correlation.correlation import CorrelationMatrix, CorrelationMethod


class CorrelationAnalysisRequest(BaseModel):
    variables: List[Timeseries] = Field(..., title="Список переменных для корреляционного анализа")
    method: CorrelationMethod = Field(default=CorrelationMethod.pearson, title="Метод расчета корреляции")

    @model_validator(mode='after')
    def validate_variables(self):
        n = len(self.variables)

        for i in range(n):
            for j in range(n):
                if i != j and self.variables[i].name == self.variables[j].name:
                    raise ValueError(f"У рядов {i + 1} и {j + 1} одинаковые названия. "
                                     "Поменяйте их чтобы функция корректно работала")
        return self

    @model_validator(mode='after')
    def validate_ts(self):
        if len(self.variables) < 2:
            raise ValueError(f"Требуется хотя бы 2 переменные для корреляции")

        return self


class CorrelationAnalysisResponse(BaseModel):
    correlation_matrix: CorrelationMatrix = Field(..., title="Квадратная корреляционная матрица")

