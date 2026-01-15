import numpy as np
import pandas as pd
from typing import Optional
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.core.application.building_model.schemas.arimax import ArimaxParams
from src.core.domain.parameter_selection.gridsearch_result.arimax import ArimaxGridsearchResult
from src.core.domain.parameter_selection.scoring.information_criteria import InformationCriteriaScoring


class ArimaGridsearch:
    def fit(
            self,
            endog: pd.Series,
            exog: Optional[pd.DataFrame],
            max_p: int = 3,
            d: int = 0,
            max_q: int = 3,
            max_P: int = 3,  # порядок сезонной авторегрессии
            max_D: int = 1,  # порядок сезонной интеграции
            max_Q: int = 3,  # порядок сезонного скользящего среднего
            m: int = 12,     # длина сезонного периода. Сложно перебирать поэтому вводится
            # информационный критерий, по которому оптимизируется модель
            scoring: InformationCriteriaScoring = InformationCriteriaScoring.aic,
    ) -> ArimaxGridsearchResult:
        self.endog = endog
        self.exog = exog
        self.scoring = scoring

        p_range = np.arange(0, max_p + 1)
        q_range = np.arange(0, max_q + 1)
        P_range = np.arange(0, max_P + 1)
        D_range = np.arange(0, max_D + 1)
        Q_range = np.arange(0, max_Q + 1)

        # Аналог itertools.product из numpy
        # Подробнее: https://stackoverflow.com/questions/4709510/itertools-product-speed-up
        #
        # Важная справочка: 5 - количество перебираемых параметров
        grid = (np.array(
                    np.meshgrid(p_range, q_range, P_range, D_range, Q_range))
                .T.reshape(-1, 5))

        scores = []
        for p_, q_, P_, D_, Q_ in grid:
            try:
                scores.append(self._score(p_, d, q_, P_, D_, Q_, m))
            except:
                continue

        results = np.array(scores)

        best_scoring = np.min(results)
        best_combination_idx = np.argmin(results)

        p, q, P, D, Q = grid[best_combination_idx]

        return ArimaxGridsearchResult(
            optimal_params=ArimaxParams(p=p, d=d, q=q, P=P, D=D, Q=Q, m=m),
            information_criteria_value=best_scoring,
            short_representation=f'SARIMAX({p},{d},{q})({P},{D},{Q})[{m}]'
        )

    def _score(self, p, d, q, P, D, Q, m):
        result = SARIMAX(
                self.endog,
                exog=self.exog,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(
            disp=False,      # не выводим в консоль логи
            method='lbfgs',  # самый быстрый численный метод оценки OLS
        )

        if self.scoring.value == InformationCriteriaScoring.aic:
            return result.aic
        elif self.scoring.value == InformationCriteriaScoring.bic:
            return result.bic
