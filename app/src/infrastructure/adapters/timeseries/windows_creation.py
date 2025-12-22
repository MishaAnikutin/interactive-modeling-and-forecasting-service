from datetime import date
from typing import Optional, List, Tuple

from src.core.domain import Timeseries


class WindowsCreation:
    @staticmethod
    def create_windows_for_series(series: Timeseries, input_size: int) -> List[Timeseries]:
        if input_size > len(series.dates):
            raise ValueError(f"Input size {input_size} is greater than series length {len(series.dates)}")

        windows = []
        for i in range(0, len(series.dates), input_size):
            windows.append(Timeseries(
                data_frequency=series.data_frequency,
                dates=series.dates[i:i + input_size],
                values=series.values[i:i + input_size],
                name=series.name,
            ))

        return windows


    def create_windows(
            self,
            exog: Optional[List[Timeseries]],
            target: Timeseries,
            input_size: int
    ) -> Tuple[
        Optional[List[List[Timeseries]]], # преобразованная матрица exog в список матриц из окон
        List[Timeseries] # преобразованный таргет в список окон
    ]:
        windows_exog = None
        if exog:
            windows_exog = []
            for variable in exog:
                windows_exog.append(self.create_windows_for_series(variable, input_size))
        windows_target = self.create_windows_for_series(target, input_size)

        return windows_exog, windows_target


    def create_windows_for_dataset(
            self,
            exog_train: Optional[List[Timeseries]],
            train_target: Timeseries,
            exog_val: Optional[List[Timeseries]],
            val_target: Timeseries,
            exog_test: Optional[List[Timeseries]],
            test_target: Timeseries,
            input_size: int,
    ) -> Tuple[
        Optional[List[List[Timeseries]]], # преобразованная матрица exog в список матриц из окон
        List[Timeseries], # преобразованный таргет в список окон
        Optional[List[List[Timeseries]]],  # преобразованная матрица exog в список матриц из окон
        List[Timeseries],  # преобразованный таргет в список окон
        Optional[List[List[Timeseries]]],  # преобразованная матрица exog в список матриц из окон
        List[Timeseries]  # преобразованный таргет в список окон
    ]:
        windows_exog_train, windows_target_train = self.create_windows(exog_train, train_target, input_size)
        windows_exog_val, windows_target_val = self.create_windows(exog_val, val_target, input_size)
        windows_exog_test, windows_target_test = self.create_windows(exog_test, test_target, input_size)
        return (
            windows_exog_train, windows_target_train,
            windows_exog_val, windows_target_val,
            windows_exog_test, windows_target_test,
        )
