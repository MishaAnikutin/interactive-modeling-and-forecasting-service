import pandas as pd
from scipy import stats

from src.core.domain.stat_test import Conclusion, SignificanceLevel
from src.core.domain.stat_test.nan_policy import NanPolicy
from src.core.domain.stat_test.alternative import Alternative
from src.core.domain.stat_test.method import Method
from src.core.domain.stat_test.mann_whitney.result import MannWhitneyResult


class MannWhitneyAdapter:
    def calculate(
            self,
            forecast: pd.Series,
            actual: pd.Series,
            significance_level: SignificanceLevel,
            use_continuity: bool,
            alternative: Alternative = Alternative.TWO_SIDED,
            method: Method = Method.AUTO,
            nan_policy: NanPolicy = NanPolicy.PROPAGATE,
    ):
        result = stats.mannwhitneyu(
            forecast, actual,
            use_continuity=use_continuity,
            alternative=alternative.value,
            method=method.value,
            nan_policy=nan_policy.value
        )

        conclusion = Conclusion.fail_to_reject if result.pvalue < significance_level else Conclusion.reject

        return MannWhitneyResult(
            p_value=result.pvalue,
            statistic=result.statistic,
            conclusion=conclusion
        )
