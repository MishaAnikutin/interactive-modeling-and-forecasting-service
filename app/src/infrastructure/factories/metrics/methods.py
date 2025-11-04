from .factory import MetricsFactory

from src.core.domain import Metric
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
class MASE(MetricServiceI):
    def apply(self, y_pred_i, y_true_i, y_pred_j, y_true_j, **kwargs) -> Metric:
        mae_i = mean_absolute_error(y_pred=y_pred_i, y_true=y_true_i)
        mae_j = mean_absolute_error(y_pred=y_pred_j, y_true=y_true_j)
        return Metric(
            type="MASE", value=mae_i / mae_j
        )


@MetricsFactory.register()
class R2(MetricServiceI):
    def apply(self, y_pred, y_true,):
        if len(y_pred) < 2:
            return Metric(
                value=None,
                type="R2"
            )
        return Metric(
            value=r2_score(y_pred=y_pred, y_true=y_true),
            type="R2"
        )


@MetricsFactory.register()
class AdjR2(MetricServiceI):
    def apply(
            self,
            y_pred,
            y_true,
            row_count: int,
            feature_count: int,
            **kwargs
    ) -> Metric:
        return Metric(
            type="Adj-R^2",
            value=1
            - (1 - r2_score(y_true, y_pred))
            * (row_count - 1)
            / (row_count - feature_count),
        )
