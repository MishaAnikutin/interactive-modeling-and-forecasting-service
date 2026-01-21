from datetime import date
from typing import Optional, List

import pandas as pd
from neuralforecast import NeuralForecast

from src.core.domain import DataFrequency, ModelMetrics, FitParams, Timeseries, ForecastResult_V2
from src.core.domain.predicting.interface import BasePredictor
from src.core.domain.serializer.interface import ModelSerializer
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter, TimeseriesTrainTestSplit
from src.infrastructure.adapters.timeseries.split_windows import WindowSplitter
from src.infrastructure.adapters.timeseries.windows_creation import WindowsCreation
from src.infrastructure.factories.metrics import MetricsFactory
from src.shared.to_panel import to_panel


class NeuralPredictAdapter_V2(BasePredictor):
    model_name = ""
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
            self,
            model_serializer: ModelSerializer,
            ts_adapter: PandasTimeseriesAdapter,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
            windows_creation: WindowsCreation,
            windows_splitter: WindowSplitter,
    ):
        self._metric_factory = metric_factory
        self._ts_spliter = ts_train_test_split
        self._ts_adapter = ts_adapter
        self._model_serializer = model_serializer
        self._windows_creation = windows_creation
        self._windows_splitter = windows_splitter

        self.target: pd.Series = None
        self.exog: Optional[pd.DataFrame] = None

        self.train_target = None
        self.train_exog = None
        self.val_target = None
        self.val_exog = None
        self.test_target = None
        self.test_exog = None

        self.nf: NeuralForecast = None
        self.model = None


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
            start_prediction_date: date,
            forecast_horizon: int
    ) -> bool:
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
            window_exog, window_target = self._windows_creation.create_window_out_for_sample(
                self.exog, self.target, self.model.input_size
            )
            nf_panel = to_panel(window_target, window_exog)
            prediction = self._predict_window(nf_panel)
            forecast_list.append(prediction)
            if self._is_last_prediction(prediction, start_predictions.index[0], forecast_horizon):
                break

        assert len(forecast_list) == forecast_horizon
        return forecast_list

    def _predict_insample(self) -> List[pd.Series]:
        forecast_list = []
        windows_exog, windows_target = self._windows_creation.create_windows(
            self.exog, self.target, self.model.input_size
        )
        if self.exog is None:
            windows_exog = [None] * len(windows_target)
        for window_target, window_exog in zip(windows_target, windows_exog):
            nf_panel = to_panel(window_target, window_exog)
            forecast_list.append(self._predict_window(nf_panel))
        return forecast_list

    def _format_forecasts(self, forecasts: List[pd.Series]) -> List[Timeseries]:
        result_forecasts = []
        for forecast in forecasts:
            ts_forecast = self._ts_adapter.from_series(forecast, self.nf.freq)
            result_forecasts.append(ts_forecast)

        return result_forecasts

    def _predict(self, forecast_horizon: int) -> List[Timeseries]:
        insample_predictions = self._predict_insample()
        out_of_sample_predictions = self._predict_out_of_sample(insample_predictions[-1], forecast_horizon)
        # срез, так как последний прогноз внутри выборки входит во вневыборочный прогноз
        forecasts = insample_predictions[:-1] + out_of_sample_predictions
        forecasts_ts = self._format_forecasts(forecasts)
        return forecasts_ts

    def _build_best_forecast(
            self,
            forecasts: List[Timeseries],
            fit_params: FitParams,
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
        assert len(train_bf) == len(self.train_target) - self.model.input_size
        val_bf = [i for i in best_forecast.dates if (fit_params.train_boundary < i <= fit_params.val_boundary)]
        assert len(val_bf) == len(self.val_target)
        test_bf = [i for i in best_forecast.dates if (i > fit_params.val_boundary)]
        assert len(test_bf) == len(self.test_target) + fit_params.forecast_horizon
        return best_forecast

    def _calculate_best_forecast_metrics(self, best_forecast: Timeseries, fit_params: FitParams) -> ModelMetrics:
        best_forecast = self._ts_adapter.to_series(best_forecast)
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

    def execute(
            self,
            model_weight: bytes,
            target: pd.Series,
            exog_df: Optional[pd.DataFrame],
            fit_params: FitParams,
            data_frequency: DataFrequency,
    ) -> ForecastResult_V2:
        fit_result = self._model_serializer.deserialize(model_weight)
        self.nf = fit_result['nf']
        self.model = fit_result['model']

        self.target = target
        self.exog = exog_df

        self._split_dataset(fit_params.train_boundary, fit_params.val_boundary)

        forecasts = self._predict(fit_params.forecast_horizon)

        # поделить этот прогноз на train/val/test
        split_forecasts = self._windows_splitter.split(
            forecasts=forecasts,
            fit_params=fit_params,
            last_date=target.index.tolist()[-1],
            freq=data_frequency
        )

        best_forecast = self._build_best_forecast(forecasts, fit_params)

        best_forecast_metrics = self._calculate_best_forecast_metrics(best_forecast, fit_params)

        return ForecastResult_V2(
            forecasts=split_forecasts,
            best_forecast=best_forecast,
            best_forecast_metrics=best_forecast_metrics,
        )
