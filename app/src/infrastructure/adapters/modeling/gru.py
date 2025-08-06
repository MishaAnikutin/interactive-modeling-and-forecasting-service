from typing import Any

import pandas as pd
from fastapi import HTTPException
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE, MSE, RMSE, MAPE
from neuralforecast.models import GRU

from logs import logger
from src.core.application.building_model.errors.lstm import LstmTrainSizeError2, LstmTrainSizeError
from src.core.application.building_model.errors.nhits import HorizonValidationError, ValSizeError, PatienceStepsError
from src.core.application.building_model.schemas.gru import GruParams, GruFitResult
from src.core.domain import FitParams, DataFrequency
from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.modeling.neural_forecast import NeuralForecastInterface
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit


class GruAdapter(NeuralForecastInterface):
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        super().__init__(metric_factory, ts_train_test_split)
        self._log = logger.getChild(self.__class__.__name__)

    @staticmethod
    def _process_params(lstm_params: GruParams) -> dict:
        loss_map = {
            "MAE": MAE,
            "MSE": MSE,
            "RMSE": RMSE,
            "MAPE": MAPE,
        }
        if lstm_params.loss not in loss_map:
            raise HTTPException(
                status_code=400,
                detail=f"Loss '{lstm_params.loss}' is not supported. Supported losses are: {list(loss_map.keys())}",
            )
        if lstm_params.valid_loss not in loss_map:
            raise HTTPException(
                status_code=400,
                detail=f"Loss '{lstm_params.valid_loss}' is not supported. Supported losses are: {list(loss_map.keys())}",
            )
        return {
            "input_size": lstm_params.input_size,
            "inference_input_size": lstm_params.inference_input_size,
            "h_train": lstm_params.h_train,
            "encoder_n_layers": lstm_params.encoder_n_layers,
            "encoder_hidden_size": lstm_params.encoder_hidden_size,
            "encoder_dropout": lstm_params.encoder_dropout,
            "decoder_hidden_size": lstm_params.decoder_hidden_size,
            "decoder_layers": lstm_params.decoder_layers,
            "recurrent": lstm_params.recurrent,
            "loss": loss_map[lstm_params.loss](),
            "valid_loss": loss_map[lstm_params.valid_loss](),
            "max_steps": lstm_params.max_steps,
            "learning_rate": lstm_params.learning_rate,
            "early_stop_patience_steps": lstm_params.early_stop_patience_steps,
            "val_check_steps": lstm_params.val_check_steps,
            "scaler_type": lstm_params.scaler_type,
        }

    def fit(
        self,
        target: pd.Series,
        exog: pd.DataFrame | None,
        gru_params: GruParams,
        fit_params: FitParams,
        data_frequency: DataFrequency
    ) -> tuple[GruFitResult, dict[str, Any]]:
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

        # валидация параметров
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

        if val_size == 0 and gru_params.early_stop_patience_steps > 0:
            raise HTTPException(
                detail=PatienceStepsError().detail,
                status_code=400,
            )

        if gru_params.input_size + h > train_target.shape[-1]:
            raise HTTPException(
                status_code=400,
                detail=LstmTrainSizeError().detail
            )

        if gru_params.recurrent and gru_params.input_size + gru_params.h_train + test_size > train_target.shape[-1]:
            raise HTTPException(
                status_code=400,
                detail=LstmTrainSizeError2().detail
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
        model = GRU(
            hist_exog_list=[exog_col for exog_col in exog.columns] if exog is not None else None,
            accelerator='cpu',
            h=h,
            **self._process_params(gru_params)
        )
        nf = NeuralForecast(models=[model], freq=data_frequency)
        nf.fit(df=train_df, val_size=val_size)
        weights = model.state_dict()

        # 4. Прогнозы -----------------------------------------------------------------
        # 4.1 train
        fcst_insample_df = nf.predict_insample()
        fcst_train = (
            fcst_insample_df.loc[fcst_insample_df['ds'].isin(train_df['ds'])]
            .drop_duplicates('ds', keep='last')
            .set_index('ds')['GRU']
        )

        # 4.2-4.3 test
        all_forecasts = nf.predict(futr_df=future_df)['GRU']
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
        )
        metrics = self._calculate_metrics(
            y_train_true=train_target,
            y_train_pred=train_predict,
            y_val_true=val_target,
            y_val_pred=validation_predict,
            y_test_true=test_target,
            y_test_pred=fcst_test,
        )
        return GruFitResult(
            forecasts=forecasts,
            model_metrics=metrics,
            weight_path='заглушка',
            model_id='заглушка',
        ), weights