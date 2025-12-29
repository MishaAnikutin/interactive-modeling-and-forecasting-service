from abc import ABC, abstractmethod
from datetime import date
from typing import Type, List

import pandas as pd
from fastapi import HTTPException
from neuralforecast import NeuralForecast
from pydantic import BaseModel

from src.core.domain import DataFrequency, FitParams, Timeseries, ModelMetrics
from src.infrastructure.adapters.modeling.interface import MlAdapterInterface
from typing import Generic, TypeVar

from src.infrastructure.adapters.modeling.neural_forecast.utils import form_train_df
from src.infrastructure.adapters.modeling_2.neural_models_errors import TrainSizeError, LSTM_GRU_TrainSizeError, \
    ValSizeError, PatienceStepsError
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit, PandasTimeseriesAdapter
from src.infrastructure.adapters.timeseries.windows_creation import WindowsCreation
from src.infrastructure.factories.metrics import MetricsFactory
from src.shared.to_panel import to_panel


TResult = TypeVar("TResult", bound=BaseModel)
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

    def _validate_params(self, hyperparameters) -> None:
        h = hyperparameters.output_size
        train_size = self.train_target.shape[0]
        val_size = self.val_target.shape[0]
        set_automatically = False
        if hyperparameters.input_size == -1:
            set_automatically = True
            hyperparameters.input_size = 3 * hyperparameters.output_size

        if val_size == 0 and hyperparameters.early_stop_patience_steps > 0:
            raise HTTPException(status_code=400, detail=PatienceStepsError().detail)

        if val_size != 0 and val_size < h:
            raise HTTPException(
                status_code=400,
                detail=ValSizeError(
                    val_size=val_size,
                    output_size=hyperparameters.output_size
                ).detail,
            )

        if hyperparameters.input_size + h > train_size:
            extra_info = (
                f". input_size выставлен автоматически в значение {hyperparameters.input_size} = (3 * output_size), "
                f"так как пользователь указал значение -1." if set_automatically else ""
            )
            raise HTTPException(
                status_code=400,
                detail=TrainSizeError(
                    input_size=hyperparameters.input_size,
                    train_size=train_size,
                    output_size=hyperparameters.output_size,
                ).detail + extra_info,
            )

        if hasattr(hyperparameters, "recurrent") and hasattr(hyperparameters, "h_train"):
            if (
                    hyperparameters.recurrent and
                    hyperparameters.input_size + hyperparameters.h_train > train_size
            ):
                extra_info = (
                    f". input_size выставлен автоматически в значение {hyperparameters.input_size} = (3 * output_size), "
                    f"так как пользователь указал значение -1." if set_automatically else ""
                )
                raise HTTPException(
                    status_code=400,
                    detail=LSTM_GRU_TrainSizeError(
                        input_size=hyperparameters.input_size,
                        train_size=train_size,
                        h_train=hyperparameters.h_train,
                    ).detail + extra_info,
                )

    def _prepare_model(self, hyperparameters: TParams) -> None:
        self.model = self.model_class(
            hist_exog_list=[exog_col for exog_col in self.exog.columns] if self.exog is not None else None,
            accelerator='cpu',
            h=hyperparameters.output_size,
            devices=1,
            **hyperparameters.model_dump()
        )

    def _fit_nf(self, data_frequency) -> None:
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

        # Создаем следующую временную метку
        freq = pd.infer_freq(self.target.index)
        last_target_index = self.target.index[-1]
        new_index = last_target_index + pd.tseries.frequencies.to_offset(freq)

        # Добавляем предсказание с правильной датой
        new_prediction_series = pd.Series(
            [first_prediction],
            index=[new_index]
        )
        self.target = pd.concat([self.target, new_prediction_series])

    def _get_forecast_dates(self, window: pd.DataFrame, output_size: int) -> pd.DatetimeIndex:
        return pd.date_range(
            start=window['ds'].iloc[-1],
            periods=output_size + 1,
            freq=self.nf.freq
        )[1:]

    def _predict_window(self, window: pd.DataFrame) -> pd.Series:
        prediction = self.nf.predict(df=window)
        prediction = prediction[self.model_name]
        prediction.index = self._get_forecast_dates(window=window, output_size=prediction.shape[0])
        return prediction

    def _is_last_prediction(
            self,
            prediction: pd.Series,
            start_prediction_date: date, forecast_horizon: int) -> bool:
        first_prediction_date = prediction.index[0]
        last_dt = pd.date_range(
            start=start_prediction_date,
            periods=forecast_horizon,
            freq=self.nf.freq
        )[-1]
        if first_prediction_date == last_dt:
            return True
        return False

    def _predict_out_of_sample(
            self,
            start_predictions: pd.Series,
            input_size: int,
            forecast_horizon: int
    ) -> List[pd.Series]:
        output_size = start_predictions.shape[0]
        if output_size >= forecast_horizon: # если мы уже получили все предсказания, то можно не делать ничего
            return [start_predictions.iloc[:forecast_horizon]]

        forecast_list = [start_predictions]
        for _ in range(forecast_horizon):
            # нужно продлить exog на 1 точку
            self._extend_exog(1)
            # нужно дополнить таргет последним предсказанием
            self._extend_target(forecast_list[-1])
            # нужно составить окно из последних input_size точек в target, exog
            window_exog, window_target = self.windows_creation.create_window_out_for_sample(
                self.exog, self.target, input_size
            )
            nf_panel = to_panel(window_target, window_exog)
            prediction = self._predict_window(nf_panel)
            forecast_list.append(prediction)
            if self._is_last_prediction(prediction, start_predictions.index[0], forecast_horizon):
                break

        assert len(forecast_list) == forecast_horizon
        return forecast_list

    def _predict_insample(self, input_size) -> List[pd.Series]:
        forecast_list = []
        windows_exog, windows_target = self.windows_creation.create_windows(self.exog, self.target, input_size)
        if self.exog is None:
            windows_exog = [None] * len(windows_target)
        for window_target, window_exog in zip(windows_target, windows_exog):
            nf_panel = to_panel(window_target, window_exog)
            forecast_list.append(self._predict_window(nf_panel))
        return forecast_list

    def _predict(self, input_size: int, forecast_horizon: int) -> List[pd.Series]:
        insample_predictions = self._predict_insample(input_size)
        out_of_sample_predictions = self._predict_out_of_sample(insample_predictions[-1], input_size, forecast_horizon)
        # срез, так как последний прогноз внутри выборки входит во вневыборочный прогноз
        forecasts = insample_predictions[:-1] + out_of_sample_predictions
        forecasts_ts = self._format_forecasts(forecasts)
        return forecasts_ts

    def _split_dataset(self, train_boundary: date, val_boundary: date) -> None:
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

    def _prepare_train_df(self) -> None:
        self.train_df = form_train_df(
            self.exog,
            self.train_target, self.val_target,
            self.exog_train, self.exog_val
        )

    def _format_forecasts(self, forecasts: List[pd.Series]) -> List[Timeseries]:
        result_forecasts = []
        for forecast in forecasts:
            ts_forecast = self.ts_adapter.from_series(forecast, self.nf.freq)
            result_forecasts.append(ts_forecast)

        return result_forecasts

    def _build_best_forecast(
            self,
            forecasts: List[Timeseries],
            fit_params: FitParams,
            input_size: int
    ) -> Timeseries:
        dates = []
        values = []
        for forecast in forecasts:
            dates.append(forecast.dates[0])
            values.append(forecast.values[0])
        best_forecast = Timeseries(
            dates=dates,
            values=values,
            data_frequency=self.nf.freq,
            name=f'Лучший прогноз {self.model_name}'
        )

        # проверки корректной работы кода
        train_bf = [i for i in best_forecast.dates if i <= fit_params.train_boundary]
        assert len(train_bf) == len(self.train_target) - input_size
        val_bf = [i for i in best_forecast.dates if (fit_params.train_boundary < i <= fit_params.val_boundary)]
        assert len(val_bf) == len(self.val_target)
        test_bf = [i for i in best_forecast.dates if (i > fit_params.val_boundary)]
        assert len(test_bf) == len(self.test_target) + fit_params.forecast_horizon
        return best_forecast

    def _calculate_best_forecast_metrics(self, best_forecast: Timeseries, fit_params: FitParams) -> ModelMetrics:
        best_forecast = self.ts_adapter.to_series(best_forecast)
        # добавь сюда визуализацию прогноза best_forecast, self.train_target, self.val_target,
        (
            forecast_train,
            forecast_val,
            forecast_test,
        ) = self._ts_spliter.split_ts(
            ts=best_forecast,
            train_boundary=fit_params.train_boundary,
            val_boundary=fit_params.val_boundary,
        )
        forecast_test = forecast_test.iloc[:self.test_target.shape[0]]
        train_target = self.train_target.loc[forecast_train.index]

        metrics = self._calculate_metrics(
            y_train_true=train_target,
            y_train_pred=forecast_train,
            y_val_true=self.val_target,
            y_val_pred=forecast_val,
            y_test_true=self.test_target,
            y_test_pred=forecast_test,
        )
        return metrics

    def fit(
            self,
            target: pd.Series,
            exog: pd.DataFrame | None,
            hyperparameters: TParams,
            fit_params: FitParams,
            data_frequency: DataFrequency
    ) -> tuple[TResult, NeuralForecast]:
        self.target = target
        self.exog = exog
        self._split_dataset(fit_params.train_boundary, fit_params.val_boundary)
        self._prepare_train_df()
        self._validate_params(hyperparameters)

        # обучить модель
        self._prepare_model(hyperparameters)
        self._fit_nf(data_frequency)

        # сделать прогноз на окнах
        forecasts = self._predict(
            input_size=hyperparameters.input_size,
            forecast_horizon=fit_params.forecast_horizon
        )
        # создать прогноз из первых точек
        best_forecast = self._build_best_forecast(forecasts, fit_params, hyperparameters.input_size)

        # получение метрик
        best_forecast_metrics = self._calculate_best_forecast_metrics(best_forecast, fit_params)

        return (
            self.result_class(
                forecasts=forecasts,
                best_forecast=best_forecast,
                best_forecast_metrics=best_forecast_metrics
            ),
            self.nf
        )