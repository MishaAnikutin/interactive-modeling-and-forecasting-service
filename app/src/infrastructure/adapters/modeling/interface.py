from abc import ABC, abstractmethod
from typing import TypeVar, Optional

import pandas as pd

from src.core.domain import ModelMetrics, Forecasts, Timeseries, FitParams, DataFrequency
from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit

ModelParams = TypeVar("ModelParams")
ModelFitResult = TypeVar("ModelFitResult")


class MlAdapterInterface(ABC):
    metrics = ()

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

    @staticmethod
    def _generate_forecasts(
        train_predict: pd.Series,
        test_predict: Optional[pd.Series],
        forecast: Optional[pd.Series]
    ) -> Forecasts:
        return Forecasts(
            train_predict=Timeseries(
                dates=train_predict.index.tolist(),
                values=train_predict.values.tolist(),
                name="Прогноз на тренировочной выборке",
            ),
            test_predict=Timeseries(
                dates=test_predict.index.tolist(),
                values=test_predict.values.tolist(),
                name="Прогноз на тестовой выборке"
            )  if train_predict is not None else None,
            forecast=Timeseries(
                dates=forecast.index.tolist(),
                values=forecast.values.tolist(),
                name="Прогноз сверх известных данных"
            ) if forecast is not None else None,
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