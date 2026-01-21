import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson


class SpuriousRegressionChecker:
    r2_threshold = 0.2
    dw_threshold = 0.1

    def check(self, y: pd.Series, X: pd.DataFrame):
        X = sm.add_constant(X)

        model = sm.OLS(y, X).fit()

        r2 = float(model.rsquared)
        dw = float(durbin_watson(model.resid))
        number_significant_coefs = sum(model.pvalues < 0.05)

        is_spurious = bool((r2 > self.r2_threshold) and (dw < self.dw_threshold))

        return r2, dw, number_significant_coefs, is_spurious
