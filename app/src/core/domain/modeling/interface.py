from abc import ABC, abstractmethod
from typing import TypeVar

import pandas as pd

from src.core.domain import ModelMetrics, Forecasts, Timeseries, FitParams, DataFrequency
from src.infrastructure.factories.metrics import MetricsFactory
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit

ModelParams = TypeVar("ModelParams")
ModelFitResult = TypeVar("ModelFitResult")


class ModelingInterface(ABC):
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
        self,
        metric_factory: MetricsFactory,
        ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        self._metric_factory = metric_factory
        self._ts_spliter = ts_train_test_split

    def _calculate_metrics(
        self,
        y_train_true: pd.Series,
        y_train_pred: pd.Series,
        y_val_true: pd.Series,
        y_val_pred: pd.Series,
        y_test_true: pd.Series,
        y_test_pred: pd.Series,
    ) -> ModelMetrics:

        train_metrics = self._metric_factory.apply(
            metrics=self.metrics, y_pred=y_train_pred, y_true=y_train_true
        )

        val_metrics = None
        if not y_val_pred.empty and not y_val_true.empty:
            val_metrics = self._metric_factory.apply(
                metrics=self.metrics, y_pred=y_val_pred, y_true=y_val_true
            )

        test_metrics = None
        if not y_test_pred.empty and not y_test_true.empty:
            test_metrics = self._metric_factory.apply(
                metrics=self.metrics, y_pred=y_test_pred, y_true=y_test_true
            )

        return ModelMetrics(
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            val_metrics=val_metrics
        )

    @staticmethod
    def _generate_forecasts(
        train_predict: pd.Series,
        validation_predict: pd.Series,
        test_predict: pd.Series,
        forecast: pd.Series,
        data_frequency: DataFrequency
    ) -> Forecasts:
        train_predict = train_predict.tail(-1)

        return Forecasts(
            train_predict=Timeseries(
                dates=train_predict.index.tolist(),
                values=train_predict.values.tolist(),
                data_frequency=data_frequency,
                name="Прогноз на обучающей выборке",
            ),
            validation_predict=Timeseries(
                dates=validation_predict.index.tolist(),
                values=validation_predict.values.tolist(),
                data_frequency=data_frequency,
                name="Прогноз на валидационной выборке"
        )  if not validation_predict.empty else None,
            test_predict=Timeseries(
                dates=test_predict.index.tolist(),
                values=test_predict.values.tolist(),
                data_frequency=data_frequency,
                name="Прогноз на тестовой выборке"
            )  if not test_predict.empty else None,
            forecast=Timeseries(
                dates=forecast.index.tolist(),
                values=forecast.values.tolist(),
                data_frequency=data_frequency,
                name="Вневыборочный прогноз"
            ) if not forecast.empty else None,
        )

    @abstractmethod
    def fit(
        self,
        target: pd.Series,
        exog: pd.DataFrame | None,
        params: ModelParams,
        fit_params: FitParams,
        data_frequency: DataFrequency,
    ) -> ModelFitResult:
        pass