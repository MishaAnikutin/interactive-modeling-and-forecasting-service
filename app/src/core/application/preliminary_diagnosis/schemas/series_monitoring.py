from typing import List
from datetime import date, datetime
from pydantic import BaseModel, Field, model_validator

from src.core.domain import Timeseries
from src.core.domain.stat_test import SignificanceLevel, Frequency2SeriesSize
from src.core.domain.stat_test.fisher import FisherTestResult
from src.core.domain.stat_test.fisher.errors import InsufficientDataError as FisherInsufficientDataError

from src.core.domain.stat_test.student import StudentTestResult
from src.core.domain.stat_test.student.errors import InsufficientDataError as StudentInsufficientDataError

from src.core.domain.stat_test.two_sigma.result import TwoSigmaTestResult
from src.core.domain.stat_test.two_sigma.errors import (
    TwoSigmaTestError,
    InsufficientDataError as TwoSigmaInsufficientDataError
)


class MonitoringTestRequest(BaseModel):
    timeseries: Timeseries
    date_boundary: date = Field(
        default=...,
        title="Граничная дата",
        description="Дата для разделения ряда или проверки аномалий"
    )

    def _get_boundary_index(self) -> int:
        """Находит индекс граничной даты в ряде или ближайшую доступную дату в пределах ряда"""
        if not self.timeseries.dates:
            raise ValueError("Ряд не содержит дат")

        min_date = min(self.timeseries.dates)
        max_date = max(self.timeseries.dates)

        if self.date_boundary < min_date:
            raise ValueError(
                f"Граничная дата {self.date_boundary} раньше начала ряда {min_date}"
            )

        if self.date_boundary > max_date:
            raise ValueError(
                f"Граничная дата {self.date_boundary} позже окончания ряда {max_date}"
            )

        # Ищем точное совпадение
        try:
            return self.timeseries.dates.index(self.date_boundary)
        except ValueError:
            # Если точной даты нет, находим ближайшую предыдущую дату
            nearest_index = None
            for i, series_date in enumerate(self.timeseries.dates):
                if series_date <= self.date_boundary:
                    nearest_index = i
                else:
                    break

            return nearest_index


class StudentTestRequest(MonitoringTestRequest):
    equal_var: bool = Field(
        True,
        title="Флаг равенства дисперсий",
        description="Если True — тест с равными дисперсиями; если False — версия Уэлча."
    )
    alpha: SignificanceLevel = Field(0.05, ge=0, le=1, title="Уровень значимости")

    @model_validator(mode='after')
    def validate_student_data(self):
        series_size = Frequency2SeriesSize.get(self.timeseries.data_frequency)

        if series_size > len(self.timeseries.values):
            raise StudentInsufficientDataError(
                'В ряде недостаточно значений для проведения теста'
                f'Нужно минимум {series_size}'
            )

        boundary_idx = self._get_boundary_index()

        points_after_split = len(self.timeseries.values) - boundary_idx
        if points_after_split < 2:
            raise StudentInsufficientDataError(
                'Недостаточно данных после граничной даты для проведения теста, '
                'нужно хотя бы 2 наблюдения'
            )

        return self


class StudentTestResponse(BaseModel):
    results: List[StudentTestResult]


class TwoSigmaTestRequest(MonitoringTestRequest):
    @model_validator(mode='after')
    def validate_two_sigma_data(self):
        series_size = Frequency2SeriesSize.get(self.timeseries.data_frequency)

        if series_size > len(self.timeseries.values):
            raise TwoSigmaInsufficientDataError(
                'В ряде недостаточно значений для проведения теста.\n'
                f'Нужно минимум {series_size}'
            )

        boundary_idx = self._get_boundary_index()

        if boundary_idx <= 3 * series_size - 1:
            raise TwoSigmaTestError(
                "Для тестирования необходимо выбрать дату не ранее чем через три года от начальной точки ряда"
            )

        return self


class TwoSigmaTestResponse(BaseModel):
    results: List[TwoSigmaTestResult]


class FisherTestRequest(MonitoringTestRequest):
    alpha: SignificanceLevel = Field(0.05, ge=0, le=1, title="Уровень значимости")

    @model_validator(mode='after')
    def validate_fisher_data(self):
        series_size = Frequency2SeriesSize.get(self.timeseries.data_frequency)

        if series_size > len(self.timeseries.values):
            raise FisherInsufficientDataError(
                'В ряде недостаточно значений для проведения теста'
                f'Нужно минимум {series_size}'
            )

        boundary_idx = self._get_boundary_index()

        points_after_boundary = len(self.timeseries.values) - boundary_idx
        if points_after_boundary <= series_size + 1:
            if boundary_idx >= (len(self.timeseries.values) - 3):
                raise FisherInsufficientDataError('В ряде недостаточно значений для проведения теста')

        return self


class FisherTestResponse(BaseModel):
    results: List[FisherTestResult]
