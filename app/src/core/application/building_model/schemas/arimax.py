from typing import List

from pydantic import BaseModel, Field, model_validator

from src.core.domain import FitParams, Coefficient, ForecastResult
from src.core.domain.model.model_data import ModelData


class ArimaxParams(BaseModel):
    p: int = Field(default=0, ge=0, le=10000, description='порядок авторегрессии')
    d: int = Field(default=0, ge=0, le=10000, description='порядок интегрирования')
    q: int = Field(default=0, ge=0, le=10000, description='порядок скользящего среднего')

    P: int = Field(default=0, ge=0, le=10000, description='порядок сезонной авторегрессии')
    D: int = Field(default=0, ge=0, le=10000, description='порядок сезонного интегрирования')
    Q: int = Field(default=0, ge=0, le=10000, description='порядок сезонного скользящего среднего')
    m: int = Field(default=0, ge=0, le=10000, description='длина сезонного периода')


class ArimaxFitRequest(BaseModel):
    model_data: ModelData = Field(default=ModelData())
    hyperparameters: ArimaxParams = Field(title='Параметры модели ARIMAX')
    fit_params: FitParams = Field(
        default=FitParams(),
        title="Общие параметры обучения",
        description="train_boundary должна быть раньше val_boundary ",
    )

    @model_validator(mode='after')
    def validate_params(self):
        start_date = min(self.model_data.dependent_variables.dates)
        end_date = max(self.model_data.dependent_variables.dates)

        if not (end_date >= self.fit_params.val_boundary >= self.fit_params.train_boundary > start_date):
            raise ValueError('Границы выборок должны быть внутри датасета, '
                             f'сейчас: '
                             f'end_date={end_date}, '
                             f'val_boundary={self.fit_params.val_boundary}, '
                             f'train_boundary={self.fit_params.train_boundary}, '
                             f'start_date={start_date}. Возможно, вы забыли указать свое значение параметра val_boundary')

        return self


class ArimaxFitResult(ForecastResult):
    coefficients: List[Coefficient] = Field(title="Список коэффициентов")

