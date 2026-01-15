import pandas as pd
from dieboldmariano import dm_test

from src.core.domain.stat_test import SignificanceLevel, Conclusion
from src.core.domain.stat_test.dm_test.loss_function import LossFunction
from src.core.domain.stat_test.dm_test.result import DmTestResult
from src.core.domain.stat_test.dm_test.variance_estimator import VarianceEstimator


class DmTestAdapter:
    def calculate(
            self,
            forecast1: pd.Series,
            forecast2: pd.Series,
            actual: pd.Series,
            significance_level: SignificanceLevel = 0.05,
            h: int = 1,
            one_sided: bool = False,
            harvey_correction: bool = True,
            variance_estimator: VarianceEstimator = VarianceEstimator.acf,
    ) -> DmTestResult:
        try:
            statistic, p_value = dm_test(
                V=actual,
                P1=forecast1,
                P2=forecast2,
                h=h, one_sided=one_sided,
                harvey_correction=harvey_correction,
                variance_estimator=variance_estimator.value,
            )
        except:
            statistic, p_value = None, 0

        conclusion = Conclusion.reject if p_value > significance_level else Conclusion.fail_to_reject

        return DmTestResult(
            statistic=statistic,
            p_value=p_value,
            conclusion=conclusion
        )
