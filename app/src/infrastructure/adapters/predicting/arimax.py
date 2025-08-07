import base64
import pandas as pd
from typing import Optional
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper

from src.infrastructure.adapters.serializer import ModelSerializer


class PredictArimaxAdapter:

    def __init__(self, model_serializer: ModelSerializer):
        self._model_serializer = model_serializer

    def execute(
            self,
            model_weight: str,
            steps: int,
            target: pd.Series,
            exog_df: Optional[pd.DataFrame]
    ) -> tuple[pd.Series, pd.Series]:
        model: SARIMAXResultsWrapper = self._model_serializer.undo_serialize(model_weight)

        model = model.apply(target, exog=exog_df)

        in_sample = model.get_prediction().predicted_mean
        out_of_sample = model.get_forecast(steps=steps, exog=exog_df.iloc[-1]).predicted_mean

        return in_sample, out_of_sample
