import pandas as pd
from typing import Optional

from fastapi import HTTPException
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAPE, RMSE, MSE, MAE
from neuralforecast.models import GRU
from neuralforecast.tsdataset import TimeSeriesDataset

from src.core.application.building_model.schemas.gru import GruParams
from src.core.domain import DataFrequency, Timeseries
from src.infrastructure.adapters.modeling.neural_forecast import future_index
from src.infrastructure.adapters.serializer import ModelSerializer


class PredictGruAdapter:
    def __init__(self, model_serializer: ModelSerializer):
        self._model_serializer = model_serializer

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
    def _future_df(
            steps: int,
            train_df: pd.Series,
            freq: DataFrequency,
    ):
        last_ds = train_df['ds'].max()
        future_dates = future_index(
            last_dt=last_ds,
            data_frequency=freq,
            periods=steps,
        )
        futr_df = pd.DataFrame({"unique_id": "ts", "ds": future_dates})

        return futr_df

    def execute(
            self,
            model_weight: bytes,
            steps: int,
            target: pd.Series,
            exog_df: Optional[pd.DataFrame],
            data_frequency: DataFrequency,
    ) -> tuple[pd.Series, pd.Series]:
        deserialized_nf: NeuralForecast = self._model_serializer.undo_serialize(model_weight)
        train_df = self._to_panel(target=target, exog=exog_df)

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
        deserialized_nf.h = steps
        deserialized_nf.models[0].h = steps

        deserialized_insample = deserialized_nf.predict_insample()
        deserialized_insample = (
            deserialized_insample
            .loc[deserialized_insample['ds']
            .isin(train_df['ds'])]
            .drop_duplicates('ds', keep='last')
            .set_index('ds')['GRU']
        )

        future_df = self._future_df(
            train_df=train_df,
            steps=steps,
            freq=data_frequency,
        )

        fcst_df = deserialized_nf.predict()
        out_of_sample = fcst_df["GRU"]
        out_of_sample.index = future_df['ds']

        return deserialized_insample, out_of_sample
