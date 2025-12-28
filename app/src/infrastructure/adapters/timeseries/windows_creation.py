from typing import Optional, List, Tuple
import pandas as pd


class WindowsCreation:
    @staticmethod
    def create_windows_for_series(series: pd.Series, input_size: int) -> List[pd.Series]:
        if input_size > len(series):
            raise ValueError(f"Input size {input_size} is greater than series length {len(series.dates)}")

        windows = []
        for i in range(len(series) - input_size + 1):
            window = series.iloc[i:i + input_size]
            windows.append(window)

        return windows

    @staticmethod
    def create_windows_for_df(df: Optional[pd.DataFrame], input_size: int) -> Optional[List[pd.DataFrame]]:
        if df is None:
            return None

        if input_size > len(df):
            raise ValueError(f"Input size {input_size} is greater than dataframe length {len(df)}")

        windows = []
        for i in range(len(df) - input_size + 1):
            window = df.iloc[i:i + input_size].copy()
            windows.append(window)

        return windows

    def create_windows(
            self,
            exog: Optional[pd.DataFrame],
            target: pd.Series,
            input_size: int
    ) -> Tuple[
        Optional[List[pd.DataFrame]], # преобразованная матрица exog в список матриц из окон
        List[pd.Series] # преобразованный таргет в список окон
    ]:
        if exog is not None and not target.index.equals(exog.index):
            raise ValueError("Индексы target и exog должны совпадать")

        windows_exog = self.create_windows_for_df(exog, input_size)
        windows_target = self.create_windows_for_series(target, input_size)

        return windows_exog, windows_target

    def create_window_out_for_sample(
            self,
            exog: Optional[pd.DataFrame],
            target: pd.Series,
            input_size: int,
    ) -> Tuple[Optional[pd.DataFrame], pd.Series]:
        windows_exog, windows_target = self.create_windows(exog, target, input_size)
        if windows_exog is None:
            return None, windows_target[-1]
        return windows_exog[-1], windows_target[-1]
