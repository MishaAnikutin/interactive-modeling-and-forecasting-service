import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.core.application.preliminary_diagnosis.schemas.statistics import StatisticsEnum
from src.core.domain.statistics import StatisticsServiceI
from .factory import StatisticsFactory


@StatisticsFactory.register(name=StatisticsEnum.mean)
class Mean(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        return np.mean(ts)


@StatisticsFactory.register(name=StatisticsEnum.median)
class Median(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        return np.median(ts)


@StatisticsFactory.register(name=StatisticsEnum.mode)
class Mode(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        series = pd.Series(ts)
        value_counts = series.value_counts()
        mode_value = value_counts.index[0]
        return mode_value


@StatisticsFactory.register(name=StatisticsEnum.variance)
class Variance(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        return np.var(ts, ddof=1)


@StatisticsFactory.register(name=StatisticsEnum.kurtosis)
class Kurtosis(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        return kurtosis(ts, bias=False, fisher=True)


@StatisticsFactory.register(name=StatisticsEnum.skewness)
class Skewness(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        return skew(ts, bias=False)


@StatisticsFactory.register(name=StatisticsEnum.coefficient_of_variation)
class VariationCoefficient(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        mean = np.mean(ts)
        std = np.std(ts, ddof=1)
        return 100 * std / mean
