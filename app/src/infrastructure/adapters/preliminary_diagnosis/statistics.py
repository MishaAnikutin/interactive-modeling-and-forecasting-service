import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from src.core.domain.preliminary_diagnosis.statistics_service import StatisticsServiceI
from src.infrastructure.adapters.preliminary_diagnosis.statistics_factory import StatisticsFactory


@StatisticsFactory.register(name="mean")
class Mean(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        return np.mean(ts)


@StatisticsFactory.register(name="median")
class Median(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        return np.median(ts)


@StatisticsFactory.register(name="mode")
class Mode(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        series = pd.Series(ts)
        value_counts = series.value_counts()
        mode_value = value_counts.index[0]
        return mode_value


@StatisticsFactory.register(name="variance")
class Variance(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        return np.var(ts, ddof=1)


@StatisticsFactory.register(name="kurtosis")
class Kurtosis(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        return kurtosis(ts, bias=False, fisher=True)


@StatisticsFactory.register(name="skewness")
class Kurtosis(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        return skew(ts, bias=False)


@StatisticsFactory.register(name="coefficient_of_variation")
class VariationCoefficient(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> float:
        mean = np.mean(ts)
        std = np.std(ts, ddof=1)
        return 100 * std / mean
