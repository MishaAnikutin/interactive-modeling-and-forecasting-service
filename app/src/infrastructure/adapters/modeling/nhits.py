import uuid
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss
from neuralforecast.models import NHITS

from logs import logger
from src.core.application.building_model.schemas.nhits import (
    NhitsParams,
    NhitsFitResult,
)

from src.core.domain import FitParams, Forecasts, Timeseries, ModelMetrics
from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit
import os
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class NhitsAdapter:
    metrics =()

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ) -> None:
        self._metric_factory = metric_factory
        self._ts_spliter = ts_train_test_split
        self._log = logger.getChild(self.__class__.__name__)

    @staticmethod
    def _to_panel(
        target: pd.Series,
        exog: pd.DataFrame | None,
        unique_id: str,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "unique_id": unique_id,
                "ds": target.index,
                "y": target.values,
            }
        )
        if exog is not None and not exog.empty:
            df = pd.concat([df.reset_index(drop=True), exog.reset_index(drop=True)], axis=1)
        return df

    @staticmethod
    def _build_nhits(params: NhitsParams) -> NHITS:
        # базовые обязательные аргументы
        nhits_kwargs: dict[str, Any] = {
            "h": params.h,
            "input_size": params.input_size,
            "max_steps": params.max_steps,
            "early_stop_patience_steps": params.early_stop_patience_steps,
            "learning_rate": params.learning_rate,
            "scaler_type": params.scaler_type,
            "accelerator": "cpu",
        }
        return NHITS(**nhits_kwargs)

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

    def fit(
            self,
            target: pd.Series,
            exog: pd.DataFrame | None,
            nhits_params: NhitsParams,
            fit_params: FitParams,
    ) -> NhitsFitResult:
        self._log.debug("Старт обучения NHiTS")

        # 1. Train / val / test split -------------------------------------------------
        (
            exog_train,
            train_target,
            exog_val,
            val_target,
            exog_test,
            test_target,
        ) = self._ts_spliter.split(
            train_boundary=fit_params.train_boundary,
            val_boundary=fit_params.val_boundary,
            target=target,
            exog=exog,
        )

        # 2. Подготовка данных --------------------------------------------------------
        uid = "ts"
        df_train_val = self._to_panel(
            pd.concat([train_target, val_target]),
            pd.concat([exog_train, exog_val]) if exog is not None else None,
            unique_id=uid,
        )
        val_size = len(val_target)

        futr_df_test = None
        if test_target is not None and len(test_target) > 0:
            futr_df_test = self._to_panel(
                test_target, exog_test, unique_id=uid
            ).drop(columns=["y"])

        # 3. Создаём и обучаем модель -------------------------------------------------
        model = self._build_nhits(
            params=nhits_params
        )
        nf = NeuralForecast(models=[model], freq=pd.infer_freq(target.index) or "D")

        nf.fit(df=df_train_val, val_size=val_size)
        self._log.info("Модель NHiTS обучена")

        # 4. Прогнозы -----------------------------------------------------------------
        # 4.1 train
        fcst_train_val: pd.Series = train_target.copy()

        # 4.2 test
        fcst_test = pd.Series(dtype=float)
        if futr_df_test is not None:
            fcst_test_df = nf.predict(futr_df=futr_df_test).set_index("ds")
            fcst_test = fcst_test_df["NHITS"]

        # 4.3 будущий out-of-sample прогноз (если надо)
        fcst_future = pd.Series(dtype=float)
        if futr_df_test is None and nhits_params.h > 0:
            fcst_future_df = nf.predict().set_index("ds")
            fcst_future = fcst_future_df["NHITS"]

        # ---------- формируем объект Forecasts -----------------------------------
        forecasts = self._generate_forecasts(
            train_predict=fcst_train_val,
            test_predict=fcst_test,
            forecast=fcst_future,
        )

        # 5. Метрики ------------------------------------------------------------------
        train_predict = fcst_train_val.loc[train_target.index]
        train_predict = train_predict.iloc[: len(train_target)]

        test_target = test_target if test_target is not None else pd.Series(dtype=float)
        test_predict = fcst_test if not fcst_test.empty else pd.Series(dtype=float)
        test_predict = test_predict.iloc[: len(test_target)]

        metrics = self._calculate_metrics(
            y_train_true=train_target,
            y_train_pred=train_predict,
            y_test_true=test_target,
            y_test_pred=test_predict,
        )

        # 6. Результат ----------------------------------------------------------------
        return NhitsFitResult(
            forecasts=forecasts,
            model_metrics=metrics,
            weight_path=str('заглушка'),
            model_id=str(uuid.uuid4()),
        )