import pandas as pd
from scipy import stats

from src.core.domain.stat_test.nan_policy import NanPolicy
from src.core.domain.stat_test.alternative import Alternative
from src.core.domain.stat_test.ttest.result import TtestResult
from src.core.domain.stat_test import Conclusion, SignificanceLevel


class TtestAdapter:
    def calculate(
            self,
            forecast: pd.Series,
            actual: pd.Series,
            significance_level: SignificanceLevel,
            alternative: Alternative = Alternative.TWO_SIDED,
            nan_policy: NanPolicy = NanPolicy.PROPAGATE,
    ) -> TtestResult:
        result = stats.ttest_rel(
            forecast, actual,
            alternative=alternative.value,
            nan_policy=nan_policy.value
        )

        conclusion = Conclusion.fail_to_reject if result.pvalue < significance_level else Conclusion.reject

        return TtestResult(
            p_value=result.pvalue,
            statistic=result.statistic,
            conclusion=conclusion
        )
