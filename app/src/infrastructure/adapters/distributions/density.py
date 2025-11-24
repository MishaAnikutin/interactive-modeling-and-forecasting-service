from typing import Literal

import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neighbors import KernelDensity
from sklearn.pipeline import Pipeline

from src.core.domain.distributions import EstimateDensity
from src.core.domain.distributions.density_estimation import EstimateDensityResult, PDF, DensityKernelResult


class DensityEstimator:
    def auto_eval(
            self,
            values: list[float],
            n_splits: int,
            step: Literal['auto'] | int,
    ) -> EstimateDensityResult:
        x = np.array(values).reshape(-1, 1)

        estimator = Pipeline([('kde', KernelDensity())])

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        bandwidth_range: np.ndarray = self._get_bandwidth_range(x)
        kernels: list[str] = [kernel.value for kernel in EstimateDensity.Kernel]
        algorithms: list[str] = [algorithm.value for algorithm in EstimateDensity.Algorithm]

        param_grid = {
            'kde__bandwidth': bandwidth_range,
            'kde__kernel': kernels,
            'kde__algorithm': algorithms
        }

        grid = GridSearchCV(
            estimator=estimator,
            cv=cv,
            param_grid=param_grid,
            n_jobs=-1
        )

        grid.fit(x)

        best_kde = grid.best_estimator_.named_steps['kde']
        grid_points = self._auto_grid(x, step)
        log_densities = best_kde.score_samples(grid_points.reshape(-1, 1))
        densities = np.exp(log_densities)

        density = PDF(x=grid_points.tolist(), y=densities.tolist())

        return EstimateDensityResult(
            density=density,
            result=DensityKernelResult(
                score=grid.best_score_,
                bandwidth=best_kde.bandwidth,
                kernel=EstimateDensity.Kernel(best_kde.kernel),
                algorithm=EstimateDensity.Algorithm(best_kde.algorithm)
            )
        )

    def eval(
            self,
            values: list[float],
            step: Literal['auto'] | int,
            kernel: EstimateDensity.Kernel = EstimateDensity.Kernel.gaussian,
            algorithm: EstimateDensity.Algorithm = EstimateDensity.Algorithm.kd_tree,
            bandwidth: float | EstimateDensity.BandwidthMethods = 1.0,
    ) -> EstimateDensityResult:
        x = np.array(values)
        grid = self._auto_grid(x, step)

        if not isinstance(bandwidth, float):
            bandwidth = bandwidth.value

        kde = KernelDensity(kernel=kernel.value, algorithm=algorithm.value, bandwidth=bandwidth)
        kde.fit(x.reshape(-1, 1))

        logdens = kde.score_samples(grid.reshape(-1, 1))
        densities = np.exp(logdens)
        score = kde.score(x.reshape(-1, 1))

        density = PDF(x=grid.tolist(), y=densities.tolist())

        return EstimateDensityResult(
            density=density,
            result=DensityKernelResult(
                score=score,
                bandwidth=bandwidth,
                kernel=EstimateDensity.Kernel(kernel),
                algorithm=EstimateDensity.Algorithm(algorithm)
            )
        )


    @staticmethod
    def _auto_grid(x: np.array, step: int | Literal['auto']) -> np.ndarray:
        if step != 'auto':
            return np.linspace(x.min() - 1.0, x.max() + 1.0, step)

        n = max(len(x), 10)
        num_points = int(np.clip(np.sqrt(n) * 50, 200, 2000))

        x_min, x_max = np.nanmin(x), np.nanmax(x)
        dx = x_max - x_min
        x_min -= dx * 0.05
        x_max += dx * 0.05

        grid = np.linspace(x_min, x_max, num_points)
        return grid

    @staticmethod
    def _get_bandwidth_range(x: np.ndarray) -> np.ndarray:
        # FIXME: подумать, как можно тут переиспользовать существующий код и надо ли вообще
        kde_scott = KernelDensity(bandwidth='scott')
        kde_scott.fit(x)
        scott_bw = kde_scott.bandwidth_

        kde_silverman = KernelDensity(bandwidth='silverman')
        kde_silverman.fit(x)
        silverman_bw = kde_silverman.bandwidth_

        base = np.mean([scott_bw, silverman_bw])

        if not np.isfinite(base) or base <= 0:
            base = np.std(x) * (len(x) ** (-1 / 5))

        # FIXME: почему их пользователь не выбирает?
        grid = np.unique(np.clip(base * np.logspace(-1.0, 1.0, 15), 1e-3, None))
        return grid
