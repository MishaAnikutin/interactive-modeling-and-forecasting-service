from abc import ABC

import pandas as pd
from fastapi import HTTPException

from src.core.domain import DataFrequency
from src.infrastructure.adapters.modeling.interface import MlAdapterInterface


class NeuralForecastInterface(MlAdapterInterface, ABC):
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
        freq_map: dict[DataFrequency, str] = {
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
                detail=f"Неподдерживаемая частотность: {data_frequency}",
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