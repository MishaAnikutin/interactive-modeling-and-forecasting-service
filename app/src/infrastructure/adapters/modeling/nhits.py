import uuid
from typing import Dict

import pandas as pd
from fastapi import HTTPException
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from logs import logger
from src.core.application.building_model.schemas.nhits import (
    NhitsParams,
    NhitsFitResult,
)

from src.core.domain import FitParams, DataFrequency
from src.infrastructure.adapters.metrics import MetricsFactory
from src.infrastructure.adapters.modeling.interface import MlAdapterInterface
from src.infrastructure.adapters.timeseries import TimeseriesTrainTestSplit


class NhitsAdapter(MlAdapterInterface):
    metrics = ("RMSE", "MAPE", "R2")

    def __init__(
            self,
            metric_factory: MetricsFactory,
            ts_train_test_split: TimeseriesTrainTestSplit,
    ):
        super().__init__(metric_factory, ts_train_test_split)
        self._log = logger.getChild(self.__class__.__name__)

    @staticmethod
    def _to_panel(
        target: pd.Series,
        exog: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "unique_id": 'ts',
                "ds": target.index,
                "y": target.values,
            }
        )
        if exog is not None and not exog.empty:
            raise NotImplementedError("Необходимо реализовать логику построения данных с экз переменными")
        return df

    @staticmethod
    def _future_index(
            last_dt: pd.Timestamp,
            data_frequency: DataFrequency,
            periods: int,
    ):
        if periods <= 0:
            return pd.DatetimeIndex([])
        freq_map: Dict[DataFrequency, str] = {
            DataFrequency.year: "Y",
            DataFrequency.month: "ME",
            DataFrequency.quart: "Q",
            DataFrequency.day: "D",
            DataFrequency.hour: "H",
            DataFrequency.minute: "T",
        }
        try:
            freq_alias = freq_map[data_frequency]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail = f"Неподдерживаемая частотность: {data_frequency}",
            )
        dr = pd.date_range(
            start=last_dt,
            periods=periods + 1,
            freq=freq_alias,
        )
        return dr[1:]

    def _future_df(
            self,
            future_size: int,
            test_target: pd.Series,
            freq: DataFrequency,
    ):
        last_known_dt = test_target.index.max()
        futr_index = self._future_index(
            last_dt=last_known_dt,
            data_frequency=freq,
            periods=future_size,
        )
        future_index = pd.concat(
            [
                test_target,
                pd.Series(index=futr_index)
            ]
        ).index

        return pd.DataFrame(
            {
                "unique_id": 'ts',
                "ds": future_index,
            }
        )

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
        test_size = test_target.shape[0]
        val_size = val_target.shape[0]
        future_size = fit_params.forecast_horizon - test_size

        if future_size < 0:
            raise HTTPException(
                detail="Горизонт прогнозирования должен быть больше или равен размеру тестовой выборки "
                f"({fit_params.forecast_horizon} < {test_size})",
                status_code=400,
            )

        if val_size != 0 and val_size < fit_params.forecast_horizon:
            raise HTTPException(
                detail="Размер валидационной выборки должен быть 0 или больше или равен горизонту прогнозирования "
                f"({val_size} < {fit_params.forecast_horizon})",
                status_code=400,
            )


        # 2. Подготовка данных --------------------------------------------------------
        train_df = self._to_panel(target=pd.concat([train_target, val_target]), exog=None)
        future_df = self._future_df(future_size=future_size, freq=fit_params.data_frequency, test_target=test_target)

        # 3. Создаём и обучаем модель -------------------------------------------------
        model = NHITS(
            accelerator='cpu',
            h=fit_params.forecast_horizon,
            input_size=fit_params.forecast_horizon * 3,
            **nhits_params.model_dump()
        )
        nf = NeuralForecast(models=[model], freq=fit_params.data_frequency)

        nf.fit(df=train_df, val_size=val_size)
        self._log.info("Модель NHiTS обучена")

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
        fcst_test = all_forecasts.iloc[:test_size].copy()
        fcst_future = all_forecasts.iloc[test_size:].copy()

        # ------------------------------------------------------------------
        # 5. Сборка результата
        forecasts = self._generate_forecasts(
            train_predict=fcst_train,
            test_predict=fcst_test,
            forecast=fcst_future,
        )
        metrics = self._calculate_metrics(
            y_train_true=train_df['y'],
            y_train_pred=fcst_train,
            y_test_true=test_target,
            y_test_pred=fcst_test,
        )
        return NhitsFitResult(
            forecasts=forecasts,
            model_metrics=metrics,
            weight_path=str('заглушка'),
            model_id=str(uuid.uuid4()),
        )