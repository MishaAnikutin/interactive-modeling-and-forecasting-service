import warnings
from arch.utility.exceptions import InfeasibleTestException
import arch.unitroot
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews, range_unit_root_test


class StationarityTesting:
    def __init__(self, ts: pd.Series, alpha=0.05):
        self.ts = ts
        self.alpha = alpha
        self.min_za_points = 50

    def dickey_fuller(self) -> bool:
        stat, p, _, _, _, _ = adfuller(self.ts.values)
        return p < self.alpha

    def kpss(self) -> bool:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            stat, p, _, _ = kpss(self.ts.values, regression="c", nlags="auto")
        return p < self.alpha

    def phillips_perron(self) -> bool:
        result = arch.unitroot.PhillipsPerron(self.ts, trend="c")
        return result.pvalue < self.alpha

    def dfgls(self) -> bool:
        try:
            max_lags = min(4, len(self.ts) // 3)
            result = arch.unitroot.DFGLS(self.ts.values, trend="c", max_lags=max_lags)
        except InfeasibleTestException:
            try:
                result = arch.unitroot.DFGLS(self.ts.values, trend="c", max_lags=0)
            except InfeasibleTestException:
                return False
        try:
            return result.pvalue < self.alpha
        except InfeasibleTestException:
            return False

    def z_andrews(self) -> bool:
        if len(self.ts) < self.min_za_points:
            return False
        stat, p, _, _, _ = zivot_andrews(
            self.ts.values,
            trim=0.15,
            regression="c"
        )
        return p < self.alpha

    def range(self) -> bool:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            stat, p, _ = range_unit_root_test(self.ts)
        return p < self.alpha


if __name__ == "__main__":
    ts = pd.Series(
        range(1, 51),
        index=pd.date_range("2022-01-01", periods=50),
    )
    stationarity_testing = StationarityTesting(ts)
    print(stationarity_testing.range())