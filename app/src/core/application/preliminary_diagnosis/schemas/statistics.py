from enum import Enum

from pydantic import BaseModel


class StatisticsEnum(str, Enum):
    mean = "mean"
    median = "median"
    mode = "mode"
    variance = "variance"
    kurtosis = "kurtosis"
    skewness = "skewness"
    coefficient_of_variation = "coefficient_of_variation"

class StatisticResult(BaseModel):
    value: float