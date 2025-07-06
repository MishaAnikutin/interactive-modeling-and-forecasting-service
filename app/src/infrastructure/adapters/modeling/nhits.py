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
            # Проверка конфликта имен
            conflict_columns = set(exog.columns) & {'unique_id', 'ds', 'y'}
            if conflict_columns:
                raise HTTPException(
                    status_code=400,
                    detail=f"Конфликт имен в экзогенных переменных: {conflict_columns}"
                )

            # Объединяем с экзогенными переменными
            df = df.set_index('ds')
            df = df.join(exog, how='left')
            df = df.reset_index()
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
            DataFrequency.year: "YE",
            DataFrequency.month: "ME",
            DataFrequency.quart: "QE",
            DataFrequency.day: "D",
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
        assert len(test_target.index.tolist()) > 0, "Похоже ты пытаешься посчитать последнюю дату от пустого массива"
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
            data_frequency: DataFrequency,
    ) -> NhitsFitResult:
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

        if val_size != 0 and val_size < h:
            raise HTTPException(
                detail="Размер валидационной выборки должен быть 0 "
                       "или больше или равен величины горизонт прогнозирования + размер тестовой выборки "
                       f"({val_size} < {h})",
                status_code=400,
            )

        if val_size == 0 and nhits_params.early_stop_patience_steps > 0:
            raise HTTPException(
                detail="Валидационная выборка должна быть не пустой, "
                       "если ранняя остановка включена (early_stop_patience_steps > 0)",
                status_code=400,
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
        future_df = self._future_df(
            future_size=fit_params.forecast_horizon,
            freq=data_frequency,
            test_target=test_target
        )

        assert future_df.shape[0] == h
        assert train_df.shape[0] == (train_target.shape[0] + val_target.shape[0])
        assert test_size + train_df.shape[0] == target.shape[0]

        # 3. Создаём и обучаем модель -------------------------------------------------
        model = NHITS(
            accelerator='cpu',
            h=h,
            input_size=h * 3,
            **nhits_params.model_dump()
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