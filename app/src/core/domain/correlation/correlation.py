from enum import StrEnum, Enum
from typing import Annotated

from pydantic import Field, BaseModel


class Correlation(BaseModel):
    value: float = Field(..., ge=-1, le=1)
    variable_1: str = Field(..., title="Первая переменная")
    variable_2: str = Field(..., title="Вторая переменная")


class CorrelationMatrix(BaseModel):
    values: list[list[Correlation]] = Field(..., title="Корреляционная матрица")


class CorrelationMethod(StrEnum):
    pearson: Annotated[str, Field('pearson', title="Корреляция Пирсона", description="Стандартный метод расчета")] = 'pearson'
    kendall: Annotated[str, Field('kendall', title="Корреляция Кендалла")] = 'kendall'
    spearman: Annotated[str, Field('spearman', title="Ранговая корреляция Спирменна")] = 'spearman'
