from abc import ABC

import pandas as pd
from fastapi import HTTPException

from src.core.domain import DataFrequency
from src.infrastructure.adapters.modeling.interface import MlAdapterInterface


def future_index(
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

    def _future_df(
            self,
            future_size: int,
            test_target: pd.Series,
            last_known_dt: pd.Timestamp,
            freq: DataFrequency,
    ):
        futr_index = future_index(
            last_dt=last_known_dt,
            data_frequency=freq,
            periods=future_size,
        )
        futr_index_expanded = (
            pd.concat([test_target, pd.Series(index=futr_index)]).index
            if test_target.shape[0] != 0 else futr_index
        )

        return pd.DataFrame(
            {
                "unique_id": 'ts',
                "ds": futr_index_expanded,
            }
        )

    @staticmethod
    def _split_predict(left_size: int, predict: pd.DataFrame):
        if left_size > 0:
            left_predict = predict.iloc[:-left_size]
            right_predict = predict.iloc[-left_size:]
        else:
            left_predict = predict.copy()
            right_predict = pd.Series()
        return left_predict, right_predict