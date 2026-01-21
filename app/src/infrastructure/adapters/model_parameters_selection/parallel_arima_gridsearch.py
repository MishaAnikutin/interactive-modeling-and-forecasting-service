import numpy as np
import pandas as pd
from typing import Optional
from multiprocessing import Pool
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.core.application.building_model.schemas.arimax import ArimaxParams
from src.core.domain.parameter_selection.gridsearch_result.arimax import ArimaxGridsearchResult, SARIMAXGridsearchUnit
from src.core.domain.parameter_selection.scoring.information_criteria import InformationCriteriaScoring
from config import Config



class ParallelArimaGridsearch:
    """
    Параллельный гридсерч для SARIMAX с использованием multiprocessing.

    Распределяет вычисление скоров по N процессам для каждой комбинации параметров.
    """

    def __init__(self):
        self.endog: Optional[pd.Series] = None
        self.exog: Optional[pd.DataFrame] = None
        self.scoring: InformationCriteriaScoring = InformationCriteriaScoring.aic

    def fit(
            self,
            endog: pd.Series,
            exog: Optional[pd.DataFrame] = None,
            max_p: int = 3,
            d: int = 0,
            max_q: int = 3,
            max_P: int = 3,
            max_D: int = 1,
            max_Q: int = 3,
            m: int = 12,
            scoring: InformationCriteriaScoring = InformationCriteriaScoring.aic,
    ) -> ArimaxGridsearchResult:
        self.endog = endog
        self.exog = exog
        self.scoring = scoring

        n_jobs = Config.PARAMETER_SELECTION_N_JOBS

        # Генерируем сетку параметров
        p_range = np.arange(0, max_p + 1)
        q_range = np.arange(0, max_q + 1)
        P_range = np.arange(0, max_P + 1)
        D_range = np.arange(0, max_D + 1)
        Q_range = np.arange(0, max_Q + 1)

        grid = (np.array(
            np.meshgrid(p_range, q_range, P_range, D_range, Q_range))
                .T.reshape(-1, 5))

        # Подготавливаем аргументы для параллельного выполнения
        task_args = [
            (p_, d, q_, P_, D_, Q_, m, endog, exog, scoring)
            for p_, q_, P_, D_, Q_ in grid
        ]

        # Выполняем параллельный расчёт скоров
        with Pool(processes=n_jobs) as pool:
            results = pool.starmap(self._score_static, task_args)

        best_result: SARIMAXGridsearchUnit = min(results, key=lambda r: r.score)

        return ArimaxGridsearchResult(
            optimal_params=best_result.params,
            information_criteria_value=best_result.score,
            short_representation=self._short_representation(best_result)
        )

    def _short_representation(self, best_result: SARIMAXGridsearchUnit) -> str:
        return (f'SARIMAX({best_result.params.p},{best_result.params.d},{best_result.params.q})'
                f'({best_result.params.P},{best_result.params.D},{best_result.params.Q})[{best_result.params.m}]')

    @staticmethod
    def _score_static(
            p: int,
            d: int,
            q: int,
            P: int,
            D: int,
            Q: int,
            m: int,
            endog: pd.Series,
            exog: Optional[pd.DataFrame],
            scoring: InformationCriteriaScoring,
    ) -> Optional[float]:
        try:
            model = SARIMAX(
                endog,
                exog=exog,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False
            )

            result = model.fit(
                disp=False,
                method='lbfgs',
                maxiter=200,  # Ограничиваем итерации для ускорения
            )

            if scoring == InformationCriteriaScoring.aic:
                score = result.aic
            else:
                score = result.bic

            return SARIMAXGridsearchUnit(
                params=ArimaxParams(
                    p=p, d=d, q=q,
                    P=P, D=D, Q=Q, m=m
                ),
                score=score
            )

        except:
            return SARIMAXGridsearchUnit(
                params=ArimaxParams(
                    p=p, d=d, q=q,
                    P=P, D=D, Q=Q, m=m
                ),
                score=float('inf')  # чтобы не влиял на поиск минимума
            )
