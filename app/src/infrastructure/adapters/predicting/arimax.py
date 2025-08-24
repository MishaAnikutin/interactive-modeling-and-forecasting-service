import pandas as pd
from typing import Optional
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from src.core.domain import DataFrequency
from src.infrastructure.adapters.predicting.interface import BasePredictor
from src.infrastructure.adapters.serializer import ModelSerializer
from src.infrastructure.adapters.timeseries import TimeseriesExtender


class PredictArimaxAdapter(BasePredictor):

    def __init__(self, model_serializer: ModelSerializer, ts_extender: TimeseriesExtender):
        self._model_serializer = model_serializer
        self._ts_extender = ts_extender

    def execute(
            self,
            model_weight: bytes,
            steps: int,
            data_frequency: DataFrequency,
            target: pd.Series,
            exog_df: Optional[pd.DataFrame]
    ) -> tuple[pd.Series, pd.Series]:
        model: SARIMAXResultsWrapper = self._model_serializer.undo_serialize(model_weight)

        model = model.apply(target, exog=exog_df)
        in_sample = model.get_prediction().predicted_mean

        if exog_df is not None:
            extended_exog = self._ts_extender.apply(
                df=exog_df, data_frequency=data_frequency, steps=steps,
            )

            out_of_sample = model.get_forecast(steps=steps, exog=extended_exog).predicted_mean
        else:
            out_of_sample = model.get_forecast(steps=steps).predicted_mean


        in_sample.name = 'Внутривыборочный прогноз'
        out_of_sample.name = 'Вневыборочный прогноз'

        return in_sample, out_of_sample
