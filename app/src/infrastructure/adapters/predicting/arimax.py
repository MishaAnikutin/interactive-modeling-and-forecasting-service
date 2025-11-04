import pandas as pd
from typing import Optional
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from src.core.domain import DataFrequency, ModelMetrics, Forecasts, FitParams, Timeseries
from src.infrastructure.adapters.predicting.interface import BasePredictor
from src.infrastructure.adapters.serializer import ModelSerializer
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter, TimeseriesTrainTestSplit, TimeseriesExtender
from src.infrastructure.factories.metrics import MetricsFactory


class PredictArimaxAdapter(BasePredictor):
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
            self,
            model_serializer: ModelSerializer,
            ts_adapter: PandasTimeseriesAdapter,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
            ts_extender: TimeseriesExtender,
    ):
        self._model_serializer = model_serializer
        self._ts_adapter = ts_adapter
        self._metric_factory = metric_factory
        self._ts_spliter = ts_train_test_split
        self._ts_extender = ts_extender

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
            ) if not validation_predict.empty else None,
            test_predict=Timeseries(
                dates=test_predict.index.tolist(),
                values=test_predict.values.tolist(),
                data_frequency=data_frequency,
                name="Прогноз на тестовой выборке"
            ) if not test_predict.empty else None,
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

        model: SARIMAXResultsWrapper = self._model_serializer.undo_serialize(model_weight)

        # Разделяем данные на train, val, test
        train_target, val_target, test_target = self._ts_spliter.split_ts(
            target, fit_params.train_boundary, fit_params.val_boundary
        )

        # Применяем модель к train данным
        model_train = model.apply(train_target, exog=exog_df.loc[train_target.index] if exog_df is not None else None)
        train_predict = model_train.get_prediction().predicted_mean

        # Применяем модель к val данным (если есть)
        validation_predict = pd.Series([], dtype=float)
        if not val_target.empty:
            model_val = model.apply(
                val_target,
                exog=exog_df.loc[val_target.index] if exog_df is not None else None
            )
            validation_predict = model_val.get_prediction().predicted_mean

        # Применяем модель к test данным (если есть)
        test_predict = pd.Series([], dtype=float)
        if not test_target.empty:
            model_test = model.apply(
                test_target,
                exog=exog_df.loc[test_target.index] if exog_df is not None else None
            )
            test_predict = model_test.get_prediction().predicted_mean

        # Получаем out-of-sample прогноз
        if exog_df is not None:
            extended_exog = self._ts_extender.apply(
                df=exog_df,
                steps=fit_params.forecast_horizon,
                data_frequency=data_frequency,
            )
            forecast = model_test.get_forecast(
                steps=fit_params.forecast_horizon,
                exog=extended_exog
            ).predicted_mean
        else:
            forecast = model_test.get_forecast(
                steps=fit_params.forecast_horizon
            ).predicted_mean

        # Создаем объекты прогнозов и метрик
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
