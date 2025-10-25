from src.core.application.preliminary_diagnosis.schemas.pp_plot import PPplotParams, PPResult
from src.infrastructure.adapters.preliminary_diagnosis.pp_plot_factory import PPplotFactory
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter
import numpy as np


class PPplotUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
        plot_factory: PPplotFactory,
    ):
        self._ts_adapter = ts_adapter
        self._plot_factory = plot_factory

    def execute(self, request: PPplotParams) -> PPResult:
        x = np.array(request.timeseries.values)
        x = x[~np.isnan(x)]
        n = len(x)
        theoretical_probs = self._plot_factory.get_theoretical_probs(
            ts=x,
            distribution=request.distribution.name
        )
        empirical_probs = np.arange(1, n + 1) / (n + 1)

        perfect_line_x = np.linspace(0, 1, 100)
        perfect_line_y = perfect_line_x

        return PPResult(
            theoretical_probs=theoretical_probs,
            empirical_probs=empirical_probs.tolist(),
            perfect_line_x=perfect_line_x.tolist(),
            perfect_line_y=perfect_line_y.tolist(),
        )

