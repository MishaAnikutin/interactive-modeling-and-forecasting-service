from abc import ABC, abstractmethod
from typing import Type

import pandas as pd
from neuralforecast import NeuralForecast
from pydantic import BaseModel

from src.core.domain import DataFrequency, FitParams
from src.infrastructure.factories.metrics import MetricsFactory
from src.infrastructure.adapters.modeling.interface import MlAdapterInterface
from src.infrastructure.adapters.modeling.neural_forecast.utils import form_train_df, form_future_df, \
    full_train_predict, full_predict
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit
from typing import Generic, Protocol, TypeVar


class ModelProtocol(Protocol):
    def __init__(self, **kwargs):
        pass


TResult = TypeVar("TResult", bound=BaseModel)
TModel = TypeVar("TModel", bound=ModelProtocol)
TParams = TypeVar("TParams", bound=BaseModel)


class NeuralForecastInterface(Generic[TParams], MlAdapterInterface, ABC):
    metrics = ("RMSE", "MAPE", "R2")
    model_name = ""

    @property
    @abstractmethod
    def model(self) -> Type[TModel]:
        pass

    @property
    @abstractmethod
    def result_class(self) -> Type[TResult]:
        pass

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        super().__init__(metric_factory, ts_train_test_split)

    @staticmethod
    def _validate_params(train_size, val_size, test_size, h, hyperparameters) -> None:
        pass

    def _get_model(self, exog: pd.DataFrame | None, hyperparameters: TParams, h: int):
        model = self.model(
            hist_exog_list=[exog_col for exog_col in exog.columns] if exog is not None else None,
            accelerator='cpu',
            h=h,
            devices=1,
            **hyperparameters.model_dump()
        )
        return model

    @staticmethod
    def _get_nf(model, data_frequency, train_df, val_size) -> NeuralForecast:
        nf = NeuralForecast(models=[model], freq=data_frequency)
        nf.fit(df=train_df, val_size=val_size)
        return nf

    def _generate_fit_result(
            self,
            train_predict,
            validation_predict,
            fcst_test,
            fcst_future,
            train_target,
            val_target,
            test_target,
            data_frequency
    ) -> TResult:
        forecasts = self._generate_forecasts(
            train_predict=train_predict,
            validation_predict=validation_predict,
            test_predict=fcst_test,
            forecast=fcst_future,
            data_frequency=data_frequency,
        )
        metrics = self._calculate_metrics(
            y_train_true=train_target,
            y_train_pred=train_predict,
            y_val_true=val_target,
            y_val_pred=validation_predict,
            y_test_true=test_target,
            y_test_pred=fcst_test,
        )
        return self.result_class(forecasts=forecasts, model_metrics=metrics)

    def _get_predict(self, nf, train_df, future_df, val_size, test_size):
        train_predict, validation_predict = full_train_predict(self.model_name, nf, train_df, val_size)
        test_predict, future_predict = full_predict(self.model_name, nf, future_df, test_size)
        return train_predict, validation_predict, test_predict, future_predict

    def fit(
            self,
            target: pd.Series,
            exog: pd.DataFrame | None,
            hyperparameters: TParams,
            fit_params: FitParams,
            data_frequency: DataFrequency
    ) -> tuple[TResult, bytes]:
        exog_train, train_target, exog_val, val_target, exog_test, test_target = self._ts_spliter.split(
            train_boundary=fit_params.train_boundary,
            val_boundary=fit_params.val_boundary,
            target=target,
            exog=exog,
        )
        train_size = train_target.shape[0]
        test_size = test_target.shape[0]
        val_size = val_target.shape[0]
        h = fit_params.forecast_horizon + test_size
        self._validate_params(train_size, val_size, test_size, h, hyperparameters)
        train_df = form_train_df(exog, train_target, val_target, exog_train, exog_val)
        future_df = form_future_df(fit_params.forecast_horizon, test_target, data_frequency)
        model = self._get_model(exog, hyperparameters, h)
        nf = self._get_nf(model, data_frequency, train_df, val_size)

        train_predict, validation_predict, test_predict, future_predict = self._get_predict(
            nf, train_df, future_df, val_size, test_size
        )

        fit_result = self._generate_fit_result(
            train_predict=train_predict,
            validation_predict=validation_predict,
            fcst_test=test_predict,
            fcst_future=future_predict,
            train_target=train_target,
            val_target=val_target,
            test_target=test_target,
            data_frequency=data_frequency
        )
        return fit_result, nf