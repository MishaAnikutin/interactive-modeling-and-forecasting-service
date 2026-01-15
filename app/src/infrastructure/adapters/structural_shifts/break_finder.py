import itertools
import numpy as np
import pandas as pd
from math import floor, inf

import statsmodels.api as sm

from src.core.application.preliminary_diagnosis.schemas.break_finder import BreakFinderResponse


class BreakFinderAdapter:
    def _break_candidates(self, n_breaks):
        """
        Функция, возвращающая генератор для потенциальных точек структурного сдвига.
        Детали: https://stackoverflow.com/a/51919157

        Keyword arguments:
        n_breaks -- число сдвигов.
        """
        gap = self.gap
        lbound = self.lbound
        ubound = self.ubound

        for nums in itertools.combinations(
            range(lbound, ubound - gap * (n_breaks - 1)), n_breaks
        ):
            yield tuple(gap * i + x for i, x in enumerate(nums))

    def _get_season(
        self,
        seasons,
    ):
        """
        Функция для расчета матрицы сезонных компонент.
        Детали: Harvey A. C. Seasonality and unobserved components models: an overview // Conference on seasonality, seasonal adjustment and their implications for short-term analysis and forecasting, Luxembourg. – 2006. – С. 10-12.

        Keyword arguments:
        seasons -- длина сезонного цикла.
        """
        n_obs = self.n_obs

        res = np.vstack([np.eye(seasons)] * (floor(n_obs / seasons) + 1))[:n_obs, :-1]
        res[(seasons - 1) :: seasons, :] = -1

        return res

    def _get_X(
        self,
        breaks,
        intercept=True,
        break_intercept=True,
        trend=False,
        break_trend=False,
    ) -> np.ndarray:
        """
        Функция для построения матрицы детерминированных компонент.

        Keyword arguments:
        intercept -- включать ли константу в перечень компонент.
        break_intercept -- разрешить сдвиг в константе.
        trend -- включать ли тренд в перечень компонент.
        break_trend -- разрешить сдвиг в тренде.
        """
        n_obs = self.n_obs

        res = np.array([]).reshape(n_obs, 0)

        if intercept:
            _di = np.ones((n_obs, 1))
            res = np.hstack([res, _di])

            if break_intercept:
                for br in breaks:
                    _di = np.vstack(
                        [
                            np.zeros((br, 1)),
                            np.ones((n_obs - br, 1)),
                        ]
                    )
                    res = np.hstack([res, _di])

        if trend:
            _dt = np.array(range(1, n_obs + 1)).reshape(n_obs, 1)
            res = np.hstack(
                [
                    res,
                    _dt,
                ]
            )

            if break_trend:
                for br in breaks:
                    _dt = np.vstack(
                        [
                            np.zeros((br, 1)),
                            np.array(range(1, n_obs - br + 1)).reshape(n_obs - br, 1),
                        ]
                    )
                    res = np.hstack([res, _dt])

        return res

    def _criterion(
        self,
        breaks,
        intercept=True,
        break_intercept=True,
        trend=False,
        break_trend=False,
        criterion: str = "ssr",
    ):
        X = self._get_X(breaks, intercept, break_intercept, trend, break_trend)
        if self.S is not None:
            X = np.hstack([X, self.S])
        return getattr(sm.OLS(self.y, X).fit(), criterion) * (
            -1 if criterion in ("rsquared", "rsquared_adj") else 1
        )

    def fit(
        self,
        endog: pd.Series,
        trim: tuple[float | int, float | int] = (0.15, 0.15),
        gap: float | int = 0.15,
        n_breaks: int = 1,
        criterion: str = "ssr",
        intercept=True,
        break_intercept=True,
        trend=False,
        break_trend=False,
        seasons=0,
    ):
        """
        Основная функция.

        Keyword arguments:
        n_breaks -- число сдвигов.
        intercept -- включать ли константу в перечень компонент.
        break_intercept -- разрешить сдвиг в константе.
        trend -- включать ли тренд в перечень компонент.
        break_trend -- разрешить сдвиг в тренде.
        """
        self.y = endog.copy()
        self.n_obs = endog.shape[0]
        self.lbound = (
                          floor(self.n_obs * trim[0]) if isinstance(trim[0], float) else trim[0]
                      ) + 1
        self.ubound = self.n_obs - (
            floor(self.n_obs * trim[1]) if isinstance(trim[1], float) else trim[1]
        )
        self.gap = floor(self.n_obs * gap) if isinstance(gap, float) else gap

        _crit = inf
        _breaks = self._break_candidates(n_breaks)
        _break = None

        assert criterion in ("ssr", "aic", "bic", "rsquared", "rsquared_adj"), (
            "Неподдерживаемый критерий!"
        )

        assert 0 <= seasons < self.n_obs, "Недопустимая длина сезона!"
        self.S = None
        if seasons > 1:
            self.S = self._get_season(seasons)

        for bs in _breaks:
            if (
                _cur := self._criterion(
                    bs,
                    intercept,
                    break_intercept,
                    trend,
                    break_trend,
                    criterion,
                )
            ) < _crit:
                _crit = _cur
                _break = bs

        if _break is None:
            raise ValueError("Не смогли найти минимум критерия!")

        # это всегда будет
        # if isinstance(self.y, pd.Series):
        #     return _break, tuple(self.y.index[_b] for _b in _break)

        break_dates = tuple(self.y.index[_b] for _b in _break)
        break_datetimes = [pd.Timestamp(date).to_pydatetime() for date in break_dates]
        return BreakFinderResponse(break_datetimes=break_datetimes)
