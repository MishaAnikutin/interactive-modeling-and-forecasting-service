from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from scipy import stats
import statistics as st

from src.core.application.preliminary_diagnosis.schemas.statistics import RusStatMetricEnum, StatisticResult
from src.core.domain.statistics import StatisticsServiceI
from .factory import StatisticsFactory

@StatisticsFactory.register(name=RusStatMetricEnum.N_OBS)
class Nobs(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=ts.size)

@StatisticsFactory.register(name=RusStatMetricEnum.MEAN)
class Mean(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=np.mean(ts))

@StatisticsFactory.register(name=RusStatMetricEnum.MEAN_CONF_INT)
class MeanConf(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value={
            'Нижний' : round(
                st.mean(ts) - stats.t.ppf(
                    1 - 0.05/2,
                    df=len(ts)-1
                ) * np.sqrt(st.variance(ts)) / np.sqrt(len(ts))
            ),
            'Верхний': round(
                st.mean(ts) + stats.t.ppf(
                    1 - 0.05/2,
                    df=len(ts)-1
                ) * np.sqrt(st.variance(ts)) / np.sqrt(len(ts))
            )
        })

@StatisticsFactory.register(name=RusStatMetricEnum.CR_BOUND_MEAN)
class CRMean(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=round(ts.var(ddof=1) / len(ts), 2))

@StatisticsFactory.register(name=RusStatMetricEnum.STD_ERR)
class StdError(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=round(stats.sem(ts, nan_policy='omit')))

@StatisticsFactory.register(name=RusStatMetricEnum.MEDIAN)
class Median(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=np.median(ts))

@StatisticsFactory.register(name=RusStatMetricEnum.STD)
class Std(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=round(st.stdev(ts)))

@StatisticsFactory.register(name=RusStatMetricEnum.GEOM_MEAN)
class GeoMean(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        value = round(st.geometric_mean(ts)) if (ts > 0).all() else None
        return StatisticResult(value=value)


@StatisticsFactory.register(name=RusStatMetricEnum.MODE)
class Mode(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        series = pd.Series(ts)
        value_counts = series.value_counts()
        mode_value = value_counts.index[0]
        return StatisticResult(value=mode_value)


@StatisticsFactory.register(name=RusStatMetricEnum.VAR)
class Variance(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=np.var(ts, ddof=1))

@StatisticsFactory.register(name=RusStatMetricEnum.VAR_CONF_INT)
class VarianceConf(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value={
            'Нижний': round(
                (len(ts)-1) * ts.var(ddof=1) / stats.chi2.ppf(1 - 0.05/2, df=len(ts)-1)
            ),
            'Верхний': round(
                (len(ts)-1) * ts.var(ddof=1) / stats.chi2.ppf(0.05/2, df=len(ts)-1)
            )
        })

@StatisticsFactory.register(name=RusStatMetricEnum.CR_BOUND_VAR)
class CRVariance(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=round(ts.var(ddof=1) / len(ts), 2))

@StatisticsFactory.register(name=RusStatMetricEnum.KURTOSIS)
class Kurtosis(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=kurtosis(ts, bias=False, fisher=True))

@StatisticsFactory.register(name=RusStatMetricEnum.SKEW)
class Skewness(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=skew(ts, bias=False))

@StatisticsFactory.register(name=RusStatMetricEnum.MIN)
class Min(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=ts.min())

@StatisticsFactory.register(name=RusStatMetricEnum.MAX)
class Max(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=ts.max())

@StatisticsFactory.register(name=RusStatMetricEnum.RANGE)
class Range(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=ts.max() - ts.min())

@StatisticsFactory.register(name=RusStatMetricEnum.SUM)
class Sum(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=ts.sum())

@StatisticsFactory.register(name=RusStatMetricEnum.Q25)
class Q25(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=np.quantile(ts, 0.25))

@StatisticsFactory.register(name=RusStatMetricEnum.Q75)
class Q75(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=np.quantile(ts, 0.75))

@StatisticsFactory.register(name=RusStatMetricEnum.LAST_Z)
class LastZ(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        if len(ts) == 0:
            return StatisticResult(value=0.0)
        z_scores = stats.zscore(ts, nan_policy='omit')
        return StatisticResult(value=float(round(z_scores[-1])))

@StatisticsFactory.register(name=RusStatMetricEnum.MEDIAN_WOLSH)
class MedianWolsh(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(
            value=round(np.median([(x + y) / 2 for x, y in combinations(ts, 2)]))
        )

@StatisticsFactory.register(name=RusStatMetricEnum.TRIMMED_MEAN)
class TrimmedMean(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(value=round(stats.trim_mean(ts, 0.1)))

@StatisticsFactory.register(name=RusStatMetricEnum.ENTROPY)
class Entropy(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        return StatisticResult(
            value=round(stats.entropy(np.histogram(ts, bins=int(np.sqrt(len(ts))))[0] / len(ts)))
        )

@StatisticsFactory.register(name=RusStatMetricEnum.VAR_COEFF)
class VariationCoefficient(StatisticsServiceI):
    def get_value(self, ts: np.ndarray) -> StatisticResult:
        mean = np.mean(ts)
        std = np.std(ts, ddof=1)
        return StatisticResult(value=100 * std / mean)
