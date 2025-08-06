import pandas as pd
from typing import Optional

from fastapi import HTTPException
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAPE, RMSE, MSE, MAE
from neuralforecast.models import GRU

from src.core.application.building_model.schemas.gru import GruParams
from src.core.domain import DataFrequency, Timeseries
from src.infrastructure.adapters.serializer import ModelSerializer


class PredictGruAdapter:
    def __init__(
            self,
            model_serializer: ModelSerializer
    ):
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
    def _get_out_of_sample_forecast(
        forecast: pd.Series
    ) -> Timeseries:
        ...

    def execute(
            self,
            model_weight: str,
            params: GruParams,
            target: pd.Series,
            exog_df: Optional[pd.DataFrame],
            data_frequency: DataFrequency,
    ) -> tuple[pd.Series, pd.Series]:

        model = GRU(
            ...
        )

        # загружаем старые веса
        old_state_dict = self._model_serializer.undo_serialize(model_weight)
        model.load_state_dict(old_state_dict)
        model.eval()

        nf = NeuralForecast(models=[model], freq=data_frequency)

        fcst_insample_df = nf.predict_insample()
        forecasts = nf.predict(futr_df=...)['GRU']


        in_sample = ...
        out_of_sample = ...

        return in_sample, out_of_sample
