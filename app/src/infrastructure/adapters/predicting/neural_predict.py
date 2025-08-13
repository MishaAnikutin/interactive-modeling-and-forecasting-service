from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd
from fastapi import HTTPException

from src.core.domain import DataFrequency
from src.infrastructure.adapters.modeling.neural_forecast import future_index


class NeuralPredictAdapter(ABC):
    @abstractmethod
    def execute(
            self,
            model_weight: bytes,
            steps: int,
            target: pd.Series,
            exog_df: Optional[pd.DataFrame],
            data_frequency: DataFrequency,
    ) -> tuple[pd.Series, pd.Series]:
        pass

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