from abc import ABC, abstractmethod
from datetime import date
from typing import Type, Optional, List, Dict

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.sphinxext.plot_directive import out_of_date
from neuralforecast import NeuralForecast
from pydantic import BaseModel

from src.core.domain import DataFrequency, FitParams, Timeseries
from src.infrastructure.adapters.modeling.interface import MlAdapterInterface, ModelParams, ModelFitResult
from typing import Generic, Protocol, TypeVar

from src.infrastructure.adapters.modeling.neural_forecast.utils import form_train_df
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit, PandasTimeseriesAdapter
from src.infrastructure.adapters.timeseries.windows_creation import WindowsCreation
from src.infrastructure.factories.metrics import MetricsFactory
from src.shared.to_panel import to_panel


class ModelProtocol(Protocol):
    def __init__(self, **kwargs):
        pass


TResult = TypeVar("TResult", bound=BaseModel)
TModel = TypeVar("TModel", bound=ModelProtocol)
TParams = TypeVar("TParams", bound=BaseModel)


class BaseNeuralForecast(Generic[TParams], MlAdapterInterface, ABC):
    metrics = ("RMSE", "MAPE", "R2")
    model_class = None
    model_name = ""

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
            windows_creation: WindowsCreation,
            ts_adapter: PandasTimeseriesAdapter,
    ):
        self.windows_creation = windows_creation
        self.ts_adapter = ts_adapter
        super().__init__(metric_factory, ts_train_test_split)
        self.nf = None
        self.model = None
        self.target: pd.Series = None
        self.exog: pd.DataFrame = None

        self.exog_train = None
        self.train_target = None
        self.exog_val = None
        self.val_target = None
        self.exog_test = None
        self.test_target = None

        self.train_df = None

    @property
    @abstractmethod
    def result_class(self) -> Type[TResult]:
        pass

    @staticmethod
    def _validate_params(train_size, val_size, test_size, h, hyperparameters) -> None:
        pass

    def _prepare_model(self, hyperparameters: TParams):
        self.model = self.model_class(
            hist_exog_list=[exog_col for exog_col in self.exog.columns] if self.exog is not None else None,
            accelerator='cpu',
            h=hyperparameters.output_size,
            devices=1,
            **hyperparameters.model_dump()
        )

    def _fit_nf(self, data_frequency) -> NeuralForecast:
        self.nf = NeuralForecast(models=[self.model], freq=data_frequency)
        self.nf.fit(df=self.train_df, val_size=self.val_target.shape[0])

    def _extend_pd_series(self, data: pd.Series, count: int) -> pd.Series:
        last_value = data.iloc[-1]
        last_index = data.index[-1]
        freq = pd.infer_freq(data.index)
        new_index = pd.date_range(start=last_index + pd.tseries.frequencies.to_offset(freq), periods=count, freq=freq)
        new_values = [last_value] * count
        new_series = pd.Series(new_values, index=new_index)
        new_data = pd.concat([data, new_series])
        return new_data

    def _extend_exog(self, output_size: int) -> None:
        if self.exog is not None:
            extended_data = {}
            for col in self.exog.columns:
                extended_series = self._extend_pd_series(self.exog[col], output_size)
                extended_data[col] = extended_series
            self.exog = pd.DataFrame(extended_data)

    def _extend_target(self, predictions: pd.Series) -> None:
        first_prediction = predictions.iloc[0]
        first_prediction_series = pd.Series(
            [first_prediction],
            index=[predictions.index[0]]
        )
        # Добавляем только одну точку к таргету
        self.target = pd.concat([self.target, first_prediction_series])

    def predict_window(self, window: pd.DataFrame) -> pd.Series:
        prediction = self.nf.predict(df=window)
        prediction = prediction[self.model_name]
        return prediction

    def predict_out_of_sample(
            self,
            start_predictions: pd.Series,
            input_size: int,
            forecast_horizon: int
    ) -> List[pd.Series]:
        output_size = start_predictions.shape[0]
        if output_size >= forecast_horizon: # если мы уже получили все предсказания, то можно не делать ничего
            return [start_predictions.iloc[:forecast_horizon]]

        forecast_list = [start_predictions]
        for i in range(forecast_horizon):
            # нужно продлить exog на 1 точку
            self._extend_exog(1)
            # нужно дополнить таргет последним предсказанием
            self._extend_target(forecast_list[-1])
            # нужно составить окно из последних input_size точек в target, exog
            window_exog, window_target = self.windows_creation.create_window_out_for_sample(
                self.exog, self.target, input_size
            )
            nf_panel = to_panel(window_target, window_exog)
            predictions = self.predict_window(nf_panel)
            forecast_list.append(predictions)

        return forecast_list

    def predict_insample(self, input_size) -> List[pd.Series]:
        forecast_list = []
        windows_exog, windows_target = self.windows_creation.create_windows(self.exog, self.target, input_size)
        if self.exog is None:
            windows_exog = [None] * len(windows_target)
        for window_target, window_exog in zip(windows_target, windows_exog):
            nf_panel = to_panel(window_target, window_exog)
            forecast_list.append(self.predict_window(nf_panel))
        return forecast_list

    def _predict(self, input_size: int, forecast_horizon: int):
        insample_predictions = self.predict_insample(input_size)
        out_of_sample_predictions = self.predict_out_of_sample(insample_predictions[-1], input_size, forecast_horizon)
        return insample_predictions + out_of_sample_predictions

    def _split_dataset(self, train_boundary: date, val_boundary: date):
        (
            self.exog_train, self.train_target,
            self.exog_val, self.val_target,
            self.exog_test, self.test_target
        ) = self._ts_spliter.split(
            train_boundary=train_boundary,
            val_boundary=val_boundary,
            target=self.target,
            exog=self.exog,
        )

    def _prepare_train_df(self):
        self.train_df = form_train_df(
            self.exog,
            self.train_target, self.val_target,
            self.exog_train, self.exog_val
        )

    def fit(
            self,
            target: pd.Series,
            exog: pd.DataFrame | None,
            hyperparameters: TParams,
            fit_params: FitParams,
            data_frequency: DataFrequency
    ) -> tuple[TResult, bytes]:
        self.target = target
        self.exog = exog
        self._split_dataset(fit_params.train_boundary, fit_params.val_boundary)
        self._prepare_train_df()

        # обучить модель
        self._prepare_model(hyperparameters)
        self._fit_nf(data_frequency)

        # сделать прогноз на окнах
        predictions = self._predict(
            input_size=hyperparameters.input_size,
            forecast_horizon=fit_params.forecast_horizon
        )

        # получение метрик
        ...
