from typing import Any

import pandas as pd
from fastapi import HTTPException
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from logs import logger
from src.core.application.building_model.errors.nhits import HorizonValidationError, ValSizeError, PatienceStepsError, \
    TrainSizeError
from src.core.application.building_model.schemas.nhits import (
    NhitsParams,
    NhitsFitResult,
)

from src.core.domain import FitParams, DataFrequency
from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.modeling.neural_forecast import NeuralForecastInterface
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit


class NhitsAdapter(NeuralForecastInterface):
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        super().__init__(metric_factory, ts_train_test_split)
        self._log = logger.getChild(self.__class__.__name__)

    def fit(
            self,
            target: pd.Series,
            exog: pd.DataFrame | None,
            hyperparameters: NhitsParams,
            fit_params: FitParams,
            data_frequency: DataFrequency,
    ) -> tuple[NhitsFitResult, dict[str, Any]]:
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
        test_size = test_target.shape[0]
        val_size = val_target.shape[0]

        h = fit_params.forecast_horizon + test_size

        if h == 0:
            raise HTTPException(
                detail=HorizonValidationError().detail,
                status_code=400,
            )

        if val_size != 0 and val_size < h:
            raise HTTPException(
                detail=ValSizeError().detail,
                status_code=400,
            )

        if val_size == 0 and hyperparameters.early_stop_patience_steps > 0:
            raise HTTPException(
                detail=PatienceStepsError().detail,
                status_code=400,
            )

        if 4 * h > train_target.shape[-1]:
            raise HTTPException(
                status_code=400,
                detail=TrainSizeError().detail
            )

        # 2. Подготовка данных --------------------------------------------------------
        if exog is not None:
            train_df = self._to_panel(
                target=pd.concat([train_target, val_target]) if val_size != 0 else train_target,
                exog=pd.concat([exog_train, exog_val]) if exog_val.shape[0] != 0 else exog_train,
            )
        else:
            train_df = self._to_panel(
                target=pd.concat([train_target, val_target]) if val_size != 0 else train_target
            )

        last_known_dt = target.index.max()
        future_df = self._future_df(
            future_size=fit_params.forecast_horizon,
            freq=data_frequency,
            test_target=test_target,
            last_known_dt=last_known_dt,
        )

        assert future_df.shape[0] == h
        assert train_df.shape[0] == (train_target.shape[0] + val_target.shape[0])
        assert test_size + train_df.shape[0] == target.shape[0]

        # 3. Создаём и обучаем модель -------------------------------------------------
        hyperparameters = hyperparameters.model_dump()
        hyperparameters['stack_types'] = ['identity'] * hyperparameters['n_stacks']
        del hyperparameters['n_stacks']

        model = NHITS(
            hist_exog_list=[exog_col for exog_col in exog.columns] if exog is not None else None,
            accelerator='cpu',
            h=h,
            input_size=h * 3,
            **hyperparameters
        )
        nf = NeuralForecast(models=[model], freq=data_frequency)
        nf.fit(df=train_df, val_size=val_size)

        # 4. Прогнозы -----------------------------------------------------------------
        # 4.1 train
        fcst_insample_df = nf.predict_insample()
        fcst_train = (
            fcst_insample_df.loc[fcst_insample_df['ds'].isin(train_df['ds'])]
            .drop_duplicates('ds', keep='last')
            .set_index('ds')['NHITS']
        )

        # 4.2-4.3 test
        all_forecasts = nf.predict(futr_df=future_df)['NHITS']
        all_forecasts.index = future_df['ds']
        if test_size > 0:
            fcst_test = all_forecasts.iloc[:test_size].copy()
            fcst_future = all_forecasts.iloc[test_size:].copy()
        else:
            fcst_test = pd.Series()
            fcst_future = all_forecasts.copy()

        # ------------------------------------------------------------------
        # 5. Сборка результата

        # делим прогноз на валидационную и обучающую часть
        if val_size > 0:
            train_predict = fcst_train.iloc[:-val_size]
            validation_predict = fcst_train.iloc[-val_size:]
        else:
            train_predict = fcst_train.copy()
            validation_predict = pd.Series()

        forecasts = self._generate_forecasts(
            train_predict=train_predict,
            validation_predict=validation_predict,
            test_predict=fcst_test,
            forecast=fcst_future,
            data_frequency=data_frequency,
        )

        # делим исходные данные на валидационную и обучающую часть
        metrics = self._calculate_metrics(
            y_train_true=train_target,
            y_train_pred=train_predict,
            y_val_true=val_target,
            y_val_pred=validation_predict,
            y_test_true=test_target,
            y_test_pred=fcst_test,
        )
        fit_result = NhitsFitResult(
            forecasts=forecasts,
            model_metrics=metrics,
        )
        return fit_result, nf