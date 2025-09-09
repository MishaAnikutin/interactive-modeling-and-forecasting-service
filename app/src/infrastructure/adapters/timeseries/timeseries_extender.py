import numpy as np
import pandas as pd

from src.core.domain import DataFrequency


class TimeseriesExtender:
    """Расширяет датасет начиная с last_date на steps периодов"""
    def apply(
            self,
            df: pd.DataFrame,
            steps: int,
            data_frequency: DataFrequency,
    ) -> pd.DataFrame:
        last_date = df.index[-1]
        last_values = df.iloc[-1].values

        index = pd.date_range(start=last_date, periods=steps, freq=data_frequency.value)

        # создает матрицу: steps строк, каждая строка - last_values
        values = np.tile(last_values, (steps, 1))

        return pd.DataFrame(values, index=index, columns=df.columns)
