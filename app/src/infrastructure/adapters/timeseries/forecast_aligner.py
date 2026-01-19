from datetime import datetime
from typing import Optional

import pandas as pd

from src.core.domain import Forecasts, Timeseries


class ForecastTargetAligner:
    """Выравнивает целевую переменную с прогнозами по выборкам"""

    def align(
            self,
            forecasts: Forecasts,
            target: Timeseries
    ) -> tuple[Timeseries, Optional[Timeseries], Optional[Timeseries]]:
        """
        Предполагает что все валидации рядов были пройдены
        и что forecasts был получен из прогноза target
        """

        train_target = self._align_by_boundaries(series=target, base=forecasts.train_predict)
        validation_target = None
        test_target = None

        if forecasts.validation_predict is not None:
            validation_target = self._align_by_boundaries(series=target, base=forecasts.validation_predict)

        if forecasts.test_predict is not None:
            test_target = self._align_by_boundaries(series=target, base=forecasts.test_predict)

        return train_target, validation_target, test_target

    @staticmethod
    def _align_by_boundaries(series: Timeseries, base: Timeseries) -> Timeseries:
        """
        Обрезает series по границам base
        """
        start = base.dates[0]
        end = base.dates[-1]

        data = [(date, value)
                for (date, value) in zip(series.dates, series.values)
                if start <= date <= end]

        return Timeseries(
            dates=[row[0] for row in data],
            values=[row[1] for row in data],
            freq=series.data_frequency,
            name=series.name
        )
