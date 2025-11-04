from pydantic import BaseModel, Field
from src.core.application.preliminary_diagnosis.schemas.qq import QQResult
from src.core.domain import Timeseries
from src.core.domain.distributions import SelectDistributionMethod, Distribution, SelectDistributionStatistics


class SelectDistRequest(BaseModel):
    timeseries: Timeseries
    method: SelectDistributionMethod = Field(description="Метод нахождения распределения")
    distribution: list[Distribution] = Field(description="Выбор распределения для перебора")
    statistics: SelectDistributionStatistics = Field(description="Выбор статистики для расчета")
    bins: int = Field(gt=0, default=20, description="Количество бинов распределений")


class SelectDistResult(BaseModel):
    name: Distribution
    score: float
    loc: float
    scale: float


class SelectDistResponse(BaseModel):
    best_dists: list[SelectDistResult]
    qq_result: QQResult
