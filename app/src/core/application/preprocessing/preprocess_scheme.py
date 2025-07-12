from typing import List, Literal, Union, Annotated
from pydantic import BaseModel, Field, ConfigDict

from src.core.domain import Timeseries

class Transformation(BaseModel):
    model_config = ConfigDict(extra="forbid")

# 1. Дифференцирование
class DiffTransformation(Transformation):
    type: Literal["diff"] = "diff"
    diff_order: int = Field(..., gt=0)

# 2. Взятие лага
class LagTransformation(Transformation):
    type: Literal["lag"] = "lag"
    lag_order: int = Field(..., gt=0)

# 3. Логарифмирование
class LogTransformation(Transformation):
    type: Literal["log"] = "log"

# 4. Потенцирование
class PowTransformation(Transformation):
    type: Literal["pow"] = "pow"
    pow_order: float = Field(..., gt=0)

# 5. Нормализация
class NormalizeTransformation(Transformation):
    type: Literal["normalize"] = "normalize"
    method: Literal["standard", "minmax"]

# 6. Экспоненциальное сглаживание (без параметров)
class ExpSmoothTransformation(Transformation):
    type: Literal["exp_smooth"] = "exp_smooth"
    span: int = Field(..., gt=0)

# 7. Преобразование Бокса-Кокса
class BoxCoxTransformation(Transformation):
    type: Literal["boxcox"] = "boxcox"
    param: float = Field(...)

# 8. Заполнение пропусков
class FillMissingTransformation(Transformation):
    type: Literal["fillna"] = "fillna"
    method: Literal["last", "backward", "mean", "median", "mode"]

# 9. Скользящее среднее
class MovingAverageTransformation(Transformation):
    type: Literal["moving_avg"] = "moving_avg"
    window: int = Field(..., gt=0)

TransformationUnion = Annotated[
    Union[
        DiffTransformation,
        LagTransformation,
        LogTransformation,
        PowTransformation,
        NormalizeTransformation,
        ExpSmoothTransformation,
        BoxCoxTransformation,
        FillMissingTransformation,
        MovingAverageTransformation
    ],
    Field(discriminator="type")
]

# Модель запроса
class PreprocessingRequest(BaseModel):
    ts: Timeseries
    transformations: List[TransformationUnion]  # Список преобразований