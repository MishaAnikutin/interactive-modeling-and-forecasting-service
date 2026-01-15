import pandas as pd
from typing import Optional, Literal, Tuple
from statsmodels.tsa.stattools import adfuller, kpss

from .factory import StationaryTestsFactory
from src.core.domain.stat_test.interface import StatTestService, TestStatistic, PValue
from src.core.domain.stat_test.supported_stat_tests import SupportedStationaryTests


@StationaryTestsFactory.register(SupportedStationaryTests.DickeyFuller)
class DickeyFuller(StatTestService):
    def calculate(
            self,
            series: pd.Series,
            # TODO:
            regression: Literal['c', 'ct', 'ctt', 'n'] = 'c',
            max_lags: Optional[int] = None,
            autolag: Optional[str] = 'AIC'
    ) -> Tuple[TestStatistic, PValue]:
        result = adfuller(
            series,
            regression=regression,
            maxlag=max_lags,
            autolag=autolag
        )

        test_statistic = result[0]
        p_value = result[1]

        return test_statistic, p_value


@StationaryTestsFactory.register(SupportedStationaryTests.KPSS)
class KPSS(StatTestService):
    def calculate(
            self,
            series: pd.Series,
            # TODO:
            regression: Literal['c', 'ct'] = 'c',
            nlags: Optional[int] = 'auto'
    ) -> Tuple[TestStatistic, PValue]:

        result = kpss(
            series,
            regression=regression,
            nlags=nlags
        )

        test_statistic = result[0]
        p_value = result[1]

        return test_statistic, p_value
