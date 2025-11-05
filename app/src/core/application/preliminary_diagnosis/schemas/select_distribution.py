from pydantic import BaseModel, Field
from src.core.domain import Timeseries
from src.core.domain.distributions import SelectDistributionMethod, Distribution, SelectDistributionStatistics, PDF, CDF


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
    best_dist_theoretical_pdf: PDF
    best_dist_theoretical_cdf: CDF

    empirical_pdf: PDF
    empirical_cdf: CDF
