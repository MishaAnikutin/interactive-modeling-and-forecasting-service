from src.shared.schemas import Metric
from .factory import MetricsFactory
from src.core.domain.metrics.metrics_service import MetricServiceI

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)


@MetricsFactory.register()
class MAPE(MetricServiceI):
    strategy = mean_absolute_percentage_error


@MetricsFactory.register()
class MAE(MetricServiceI):
    strategy = mean_absolute_error


@MetricsFactory.register()
class RMSE(MetricServiceI):
    strategy = root_mean_squared_error


@MetricsFactory.register()
class MSE(MetricServiceI):
    strategy = mean_squared_error


@MetricsFactory.register()
class R2(MetricServiceI):
    strategy = r2_score


@MetricsFactory.register()
class AdjR2(MetricServiceI):
    def apply(self, row_count: int, feature_count: int) -> Metric:
        return Metric(
            type="Adj-R^2",
            value=1
            - (1 - r2_score(self._y_pred, self._y_true))
            * (row_count - 1)
            / (row_count - feature_count),
        )
