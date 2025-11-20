from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from distfit import distfit

from src.core.application.preliminary_diagnosis.schemas.select_distribution import SelectDistRequest, SelectDistResult, \
    Distribution


class DistFit:
    def calculate(self, request: SelectDistRequest) -> SelectDistResult:
        dmodel = distfit(
            method=request.method.value,
            distr=[dist.value for dist in request.distribution],
            stats=request.statistics.value,
            bins=request.bins,
            todf=True
        )

        print(dmodel)

        dmodel.fit_transform(np.array(request.timeseries.values))

        print(dmodel.summary)
        df = dmodel.summary.copy()

        df_ranked = df.sort_values('score', ascending=True)
        top5 = df_ranked[['name', 'score', 'loc', 'scale']].head(5)

        results: list[SelectDistResult] = list()

        for (name, score, loc, scale) in top5.values:
            results.append(SelectDistResult(
                name=Distribution(name),
                score=score, loc=loc, scale=scale
            ))

        return results
