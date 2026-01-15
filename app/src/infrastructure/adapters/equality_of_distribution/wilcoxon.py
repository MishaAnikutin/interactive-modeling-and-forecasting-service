import pandas as pd
from scipy import stats

from src.core.domain.stat_test.method import Method
from src.core.domain.stat_test.nan_policy import NanPolicy
from src.core.domain.stat_test.alternative import Alternative
from src.core.domain.stat_test.ttest.result import TtestResult
from src.core.domain.stat_test import Conclusion, SignificanceLevel
from src.core.domain.stat_test.wilcoxon.params import ZeroMethod


class WilcoxonAdapter:
    def calculate(
            self,
            forecast: pd.Series,
            actual: pd.Series,
            significance_level: SignificanceLevel = 0.05,
            correction: bool = False,
            alternative: Alternative = Alternative.TWO_SIDED,
            nan_policy: NanPolicy = NanPolicy.PROPAGATE,
            method: Method = Method.AUTO,
            zero_method: ZeroMethod = ZeroMethod.wilcox
    ) -> TtestResult:
        result = stats.wilcoxon(
            forecast, actual,
            alternative=alternative.value,
            nan_policy=nan_policy.value,
            correction=correction,
            method=method.value,
            zero_method=zero_method.value,
        )

        conclusion = Conclusion.fail_to_reject if result.pvalue < significance_level else Conclusion.reject

        return TtestResult(
            p_value=result.pvalue,
            statistic=result.statistic,
            conclusion=conclusion
        )
