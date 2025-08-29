from typing import Optional

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.tsdataset import TimeSeriesDataset

from src.core.domain import DataFrequency, ModelMetrics, Forecasts, FitParams, Timeseries
from src.infrastructure.adapters.predicting.interface import BasePredictor
from src.infrastructure.adapters.serializer import ModelSerializer
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter, TimeseriesTrainTestSplit
from src.infrastructure.adapters.metrics import MetricsFactory
from src.shared.future_dates import future_dates
from src.shared.to_panel import to_panel


class NeuralPredictAdapter(BasePredictor):
    model_name = ""
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
            self,
            model_serializer: ModelSerializer,
            ts_adapter: PandasTimeseriesAdapter,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        self._metric_factory = metric_factory
        self._ts_spliter = ts_train_test_split
        self._ts_adapter = ts_adapter
        self._model_serializer = model_serializer

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

    def execute(
            self,
            model_weight: bytes,
            target: pd.Series,
            exog_df: Optional[pd.DataFrame],
            fit_params: FitParams,
            data_frequency: DataFrequency,
    ) -> tuple[Forecasts, ModelMetrics]:
        deserialized_nf: NeuralForecast = self._model_serializer.undo_serialize(model_weight)
        train_df = to_panel(target=target, exog=exog_df)

        dataset, uids, _, ds = TimeSeriesDataset.from_df(
            df=train_df,
            static_df=None,
            id_col='unique_id',
            time_col='ds',
            target_col='y'
        )
        deserialized_nf.dataset = dataset
        deserialized_nf.uids = uids
        deserialized_nf.ds = ds
        deserialized_nf.h = fit_params.forecast_horizon
        deserialized_nf.models[0].h = fit_params.forecast_horizon

        deserialized_insample = deserialized_nf.predict_insample()
        deserialized_insample = (
            deserialized_insample
            .loc[deserialized_insample['ds']
            .isin(train_df['ds'])]
            .drop_duplicates('ds', keep='last')
            .set_index('ds')[self.model_name]
        )

        future_df = self._future_df(
            train_df=train_df,
            steps=fit_params.forecast_horizon,
            freq=data_frequency,
        )

        fcst_df = deserialized_nf.predict()
        forecast = fcst_df[self.model_name]
        forecast.index = future_df['ds']

        train_target, val_target, test_target = self._ts_spliter.split_ts(
            target, fit_params.train_boundary, fit_params.val_boundary
        )
        train_predict, validation_predict, test_predict = self._ts_spliter.split_ts(
            deserialized_insample, fit_params.train_boundary, fit_params.val_boundary
        )

        assert train_predict.shape[0] == train_target.shape[0]
        assert validation_predict.shape[0] == val_target.shape[0]
        assert test_predict.shape[0] == test_target.shape[0]
        assert forecast.shape[0] == fit_params.forecast_horizon

        forecasts = self._generate_forecasts(
            train_predict=train_predict,
            validation_predict=validation_predict,
            test_predict=test_predict,
            forecast=forecast,
            data_frequency=data_frequency,
        )

        metrics = self._calculate_metrics(
            y_train_true=train_target,
            y_train_pred=train_predict,
            y_val_true=val_target,
            y_val_pred=validation_predict,
            y_test_true=test_target,
            y_test_pred=test_predict,
        )

        return forecasts, metrics

    @staticmethod
    def _future_df(
            steps: int,
            train_df: pd.DataFrame,
            freq: DataFrequency,
    ):
        last_ds = train_df['ds'].max()
        ds = future_dates(
            last_dt=last_ds,
            data_frequency=freq,
            periods=steps,
        )
        futr_df = pd.DataFrame({"unique_id": "ts", "ds": ds})

        return futr_df