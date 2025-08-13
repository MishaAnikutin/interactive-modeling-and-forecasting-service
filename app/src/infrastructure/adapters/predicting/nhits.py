from typing import Optional

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.tsdataset import TimeSeriesDataset

from src.core.domain import DataFrequency
from src.infrastructure.adapters.predicting.neural_predict import NeuralPredictAdapter
from src.infrastructure.adapters.serializer import ModelSerializer


class PredictNhitsAdapter(NeuralPredictAdapter):
    def __init__(self, model_serializer: ModelSerializer):
        self._model_serializer = model_serializer

    def execute(
        self,
        model_weight: bytes,
        steps: int,
        target: pd.Series,
        exog_df: Optional[pd.DataFrame],
        data_frequency: DataFrequency
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
            .set_index('ds')['NHITS']
        )

        future_df = self._future_df(
            train_df=train_df,
            steps=steps,
            freq=data_frequency,
        )

        fcst_df = deserialized_nf.predict()
        out_of_sample = fcst_df["NHITS"]
        out_of_sample.index = future_df['ds']

        return deserialized_insample, out_of_sample

