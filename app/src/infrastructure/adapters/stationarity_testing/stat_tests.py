import warnings
from typing import Optional

from arch.utility.exceptions import InfeasibleTestException
import arch.unitroot
from starlette.exceptions import HTTPException
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import adfuller, kpss, zivot_andrews, range_unit_root_test

from src.core.application.preliminary_diagnosis.schemas.stats_test import StatTestResult
from src.core.domain.statistical_tests.stationarity_testing_service import StationarityTestingService
from src.infrastructure.adapters.stationarity_testing.factory import StationarityTestsFactory


@StationarityTestsFactory.register()
class DickeyFuller(StationarityTestingService):
    def apply(self) -> StatTestResult:
        stat, p, _, _, _, _ = adfuller(self.ts.values)
        return StatTestResult(
            p_value=p,
            stat_value=stat,
            test_name=self.__class__.__name__,
            alpha=self.alpha,
        )


@StationarityTestsFactory.register()
class KPSS(StationarityTestingService):
    def apply(self) -> StatTestResult:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            stat, p, _, _ = kpss(self.ts.values, regression="c", nlags="auto")
        return StatTestResult(
            p_value=p,
            stat_value=stat,
            test_name=self.__class__.__name__,
            alpha=self.alpha,
        )

@StationarityTestsFactory.register()
class PhillipsPerron(StationarityTestingService):
    def apply(self) -> StatTestResult:
        result = arch.unitroot.PhillipsPerron(self.ts, trend="c")
        return StatTestResult(
            p_value=result.pvalue,
            stat_value=result.stat,
            test_name=self.__class__.__name__,
            alpha=self.alpha,
        )

@StationarityTestsFactory.register()
class DFGLS(StationarityTestingService):
    def apply(self) -> Optional[StatTestResult]:
        try:
            max_lags = min(4, len(self.ts) // 3)
            result = arch.unitroot.DFGLS(self.ts.values, trend="c", max_lags=max_lags)
        except InfeasibleTestException:
            try:
                result = arch.unitroot.DFGLS(self.ts.values, trend="c", max_lags=0)
            except InfeasibleTestException:
                raise HTTPException(status_code=404, detail="Для данного ряда тест не применим.")
        try:
            return StatTestResult(
                p_value=result.pvalue,
                stat_value=result.stat,
                test_name=self.__class__.__name__,
                alpha=self.alpha,
            )
        except InfeasibleTestException:
            raise HTTPException(status_code=404, detail="Для данного ряда тест не применим.")

@StationarityTestsFactory.register()
class ZivotAndrews(StationarityTestingService):
    def apply(self) -> Optional[StatTestResult]:
        if len(self.ts) < 50:
            raise HTTPException(status_code=404, detail="Для данного ряда тест не применим.")
        stat, p, _, _, _ = zivot_andrews(
            self.ts.values,
            trim=0.15,
            regression="c"
        )
        return StatTestResult(
            p_value=p,
            stat_value=stat,
            test_name=self.__class__.__name__,
            alpha=self.alpha,
        )

@StationarityTestsFactory.register()
class Range(StationarityTestingService):
    def apply(self) -> Optional[StatTestResult]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=InterpolationWarning)
            stat, p, _ = range_unit_root_test(self.ts)
        return StatTestResult(
            p_value=p,
            stat_value=stat,
            test_name=self.__class__.__name__,
            alpha=self.alpha,
        )