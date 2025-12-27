from abc import ABC, abstractmethod
from datetime import date
from typing import Type, Optional, List, Dict

import pandas as pd
from matplotlib import pyplot as plt
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
        return None

    def _extend_target(self, predictions: pd.Series) -> None:
        pass

    def predict_out_of_sample(self, start_predictions: pd.Series, forecast_horizon: int) -> List[Timeseries]:
        output_size = start_predictions.shape[0]
        if output_size >= forecast_horizon: # если мы уже получили все предсказания, то можно не делать ничего
            return [start_predictions.iloc[:forecast_horizon]] # TODO: здесь понять какую структуру данных использовать

        # общее хранилище для предсказаний
        predictions_list = [start_predictions]
        # нужно получить все последние точки и продлить их на forecast_horizon значений
        for i in range(forecast_horizon):
            self._extend_exog(output_size)
            self._extend_target(predictions_list[-1])
            predictions = ... # TODO: логика получения прогноза для окна
            predictions_list.append(predictions)

        return predictions_list

    def predict_sample(self, input_size: int, output_size: int, forecast_horizon: int) -> List[Timeseries]: # TODO: сюда вставить структуру данных для прогнозов
        # train прогнозы
        exog_train = self.ts_adapter.from_dataframe_to_list(self.exog_train, freq="ME") if self.exog_train is not None else None
        train_target = self.ts_adapter.from_series(self.train_target, freq="ME") # TODO: переработать говнокод если найдешь
        windows_exog, windows_target = self.windows_creation.create_windows(
            exog_train,
            train_target,
            input_size
        )
        forecast_list = []
        if windows_exog is None:
            for window_target in windows_target: # TODO: вынести логику прогноза на окне в отдельный метод
                target_df = self.ts_adapter.to_dataframe(window_target)
                nf_panel = to_panel(target_df, None)
                forecast = self.nf.predict(df=nf_panel)
                forecast_list.append(forecast)
        else:
            for window_exog, window_target in zip(windows_exog, windows_target):
                target_df = self.ts_adapter.to_dataframe(window_target)
                exog_df = self.ts_adapter.to_dataframe_from_list(window_exog)
                nf_panel = to_panel(target_df, exog_df)
                forecast = self.nf.predict(df=nf_panel)
                forecast_list.append(forecast)

        # val прогнозы






        # через матплотлиб построим график
        # fig, ax = plt.subplots(figsize=(12, 6))
        #
        # colors = ['red', 'blue', 'green', 'orange', 'purple',
        #           'brown', 'pink', 'gray', 'olive', 'cyan']
        #
        # last_date = forecast_list[0]['ds'].iloc[-1]
        # color = 'red'
        # ax.plot(forecast_list[0]['ds'], forecast_list[0]['NHITS'],
        #         color=color, linewidth=2, alpha=0.7,
        #         label=f'Прогноз {0 + 1}')
        #
        # for idx, forecast_df in enumerate(forecast_list[1:]):
        #     first_date_forecast = forecast_df['ds'].iloc[0]
        #     last_date_forecast = forecast_df['ds'].iloc[-1]
        #
        #
        #     last_date = last_date_forecast
        #
        #     color = 'red'
        #     ax.plot(forecast_df['ds'], forecast_df['NHITS'],
        #             color=color, linewidth=2, alpha=0.7,
        #             label=f'Прогноз {idx + 1}')
        #
        # ax.plot(self.train_target, color='blue', label='Настоящий ряд', linewidth=2, alpha=0.7,)
        # ax.set_xlabel('Дата')
        # ax.set_ylabel('NHITS')
        # ax.set_title('Все прогнозы')
        # ax.legend()
        # ax.grid(True, alpha=0.3)
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.show()

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

        # обучить модель на полном наборе данных (train+val)
        self._prepare_model(hyperparameters)
        self._fit_nf(data_frequency)

        # сделать прогноз на окнах
        self.predict(
            input_size=hyperparameters.input_size,
            output_size=hyperparameters.output_size,
            forecast_horizon=fit_params.forecast_horizon
        )

        # разбить на окна для прогнозов


        # получение прогнозов по окнам


        # получение метрик

