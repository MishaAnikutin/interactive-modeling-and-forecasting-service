from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

from distfit import distfit

from src.core.application.preliminary_diagnosis.schemas.select_distribution import SelectDistRequest, SelectDistResult, \
    Distribution


class DistFit:

    # FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME
    def calculate(self, request: SelectDistRequest) -> SelectDistResult:
        dmodel = distfit(
            method=request.method.value,
            distr=[dist.value for dist in request.distribution],
            stats=request.statistics.value,
            bins=request.bins,
            todf=True
        )

        dmodel.fit_transform(np.array(request.timeseries.values))

        df = dmodel.summary.copy()
        df_ranked = df.sort_values('score', ascending=True)
        top5 = df_ranked[['name', 'score', 'loc', 'scale']].head(5)

        results: list[SelectDistResult] = list()

        for (name, score, loc, scale) in top5.values:
            results.append(SelectDistResult(
                name=Distribution(name),
                score=score, loc=loc, scale=scale
            ))

        # best_dist, best_params = extract_dist_and_params(top5.iloc[0])

        return results


# FIXME FIXME FIXME FIXME FIXME FIXME FIXME FIXME
# господи какой же хуевый код написал вова я в ахуе
def extract_dist_and_params(row: pd.Series):
    """
    Возвращает (dist, params) для scipy.stats из строки summary:
    - если есть row['params'] -> используем его напрямую (shape..., loc, scale)
    - иначе собираем из row['arg'] (tuple) + loc + scale
    - fallback: arg1..arg4 + loc + scale
    """

    name = str(row['name'])
    dist = getattr(stats, name, None)
    if dist is None:
        raise ValueError(f"Не найдено scipy-распределение '{name}'")

    # 1) Полный набор уже есть
    if 'params' in row and isinstance(row['params'], (tuple, list)):
        params = tuple(row['params'])
        return dist, params

    # 2) Есть arg как tuple + loc/scale
    if 'arg' in row and isinstance(row['arg'], (tuple, list)):
        loc   = float(row.get('loc', 0.0))
        scale = float(row.get('scale', 1.0))
        params = tuple(row['arg']) + (loc, scale)
        return dist, params

    # 3) Рассыпанные arg1..arg4 + loc/scale
    shape_cols = [c for c in ['arg1','arg2','arg3','arg4'] if c in row.index and pd.notna(row[c])]
    shape_args = [float(row[c]) for c in shape_cols]
    loc   = float(row.get('loc', 0.0))
    scale = float(row.get('scale', 1.0))
    params = tuple(shape_args + [loc, scale])
    return dist, params
