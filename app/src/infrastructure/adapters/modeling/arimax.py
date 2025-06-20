import pandas as pd
from typing import List
import statsmodels.api as sm
from logs import logger

from src.core.domain import FitParams, ModelMetrics, Forecasts, Coefficient, Timeseries
from src.core.application.building_model.schemas.arimax import ArimaxParams, ArimaxFitResult

from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit
from src.infrastructure.adapters.metrics import MetricsFactory


class ArimaxAdapter:
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
        self,
        metric_factory: MetricsFactory,
        ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        self._metric_factory = metric_factory
        self._ts_spliter = ts_train_test_split
        self._log = logger.getChild(self.__class__.__name__)

    def fit(
        self,
        target: pd.Series,
        exog: pd.DataFrame | None,
        arimax_params: ArimaxParams,
        fit_params: FitParams,
    ) -> ArimaxFitResult:
        self._log.debug(
            "Старт обучения ARIMAX",
            extra = {"target_shape": target.shape, "exog_shape": getattr(exog, "shape", None)},
        )

        # Делим выборку на обучающую и тестовую
        exog_train, train_target, exog_test, test_target = self._ts_spliter.split(
            train_boundary=fit_params.train_boundary, target=target, exog=exog
        )

        # Создаем и обучаем модель
        model = sm.tsa.ARIMA(
            endog=train_target,
            exog=exog_train,
            order=(arimax_params.p, arimax_params.d, arimax_params.q),
        )
        results = model.fit()

        self._log.info("Модель обучена", extra={"aic": results.aic, "bic": results.bic})

        # Строим прогноз на обучающей выборке
        train_predict = results.get_prediction().predicted_mean

        # Тестовой выборке
        test_predict = results.get_forecast(
            steps=len(test_target), exog=exog_test
        ).predicted_mean

        # и вневыборочный прогноз (если нет экзогенных переменных)
        forecast = (
            None
            if exog is not None
            else results.forecast(steps=fit_params.forecast_horizon)
        )

        # Собираем все ответы вместе
        forecasts = self._generate_forecasts(
            train_predict=train_predict, test_predict=test_predict, forecast=forecast
        )

        metrics = self._calculate_metrics(
            y_train_true=train_target,
            y_train_pred=train_predict,
            y_test_true=test_target,
            y_test_pred=test_predict,
        )

        coefficients = self._parse_coefficients(results)

        return ArimaxFitResult(
            coefficients=coefficients,
            metrics=metrics,
            forecasts=forecasts,
        )

    def _parse_coefficients(self, results) -> List[Coefficient]:
        return [
            Coefficient(name=name, value=value, p_value=results.pvalues[name])
            for name, value in results.params.items()
        ]

    def _calculate_metrics(
        self,
        y_train_true: pd.Series,
        y_train_pred: pd.Series,
        y_test_true: pd.Series,
        y_test_pred: pd.Series,
    ) -> ModelMetrics:

        train_metrics = self._metric_factory.apply(
            metrics=self.metrics, y_pred=y_train_pred, y_true=y_train_true
        )
        test_metrics = self._metric_factory.apply(
            metrics=self.metrics, y_pred=y_test_pred, y_true=y_test_true
        )

        return ModelMetrics(train_metrics=train_metrics, test_metrics=test_metrics)

    def _generate_forecasts(
        self, train_predict: pd.Series, test_predict: pd.Series, forecast: pd.Series
    ) -> Forecasts:
        return Forecasts(
            train_predict=Timeseries(
                dates=train_predict.index.tolist(),
                values=train_predict.values.tolist(),
            ),
            test_predict=Timeseries(
                dates=test_predict.index.tolist(),
                values=test_predict.values.tolist(),
            ),
            forecast=Timeseries(
                dates=forecast.index.tolist(),
                values=forecast.values.tolist(),
            ),
        )
