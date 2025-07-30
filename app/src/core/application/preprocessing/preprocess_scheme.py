from typing import List, Literal, Union, Annotated
from pydantic import BaseModel, Field, ConfigDict, model_validator

from src.core.domain import Timeseries
from src.shared.utils import validate_float_param


class Transformation(BaseModel):
    model_config = ConfigDict(extra="forbid")


# 1. Дифференцирование
class DiffTransformation(Transformation):
    type: Literal["diff"] = "diff"
    diff_order: int = Field(
        title="Периоды сдвига для вычисления разности (допустимы отрицательные значения)",
        description="- число наблюдений <= diff_order <= число наблюдений"
    )


# 2. Взятие лага
class LagTransformation(Transformation):
    type: Literal["lag"] = "lag"
    lag_order: int = Field(
        title="Периоды сдвига для вычисления лага (допустимы отрицательные значения)",
        description="- число наблюдений <= diff_order <= число наблюдений"
    )

# 3. Логарифмирование

class LogTransformation(Transformation):
    type: Literal["log"] = "log"


# 4. Потенцирование
class PowTransformation(Transformation):
    type: Literal["pow"] = "pow"
    pow_order: float = Field(..., gt=0, title="Показатель степени", le=100)


class MinMaxTransformation(Transformation):
    type: Literal["minmax"] = "minmax"


class StandardTransformation(Transformation):
    type: Literal["standard"] = "standard"


# 6. Экспоненциальное сглаживание (без параметров)
class ExpSmoothTransformation(Transformation):
    type: Literal["exp_smooth"] = "exp_smooth"
    span: int = Field(gt=0, title="Период окна экспоненциального сглаживания", le=1000)


# 7. Преобразование Бокса-Кокса
class BoxCoxTransformation(Transformation):
    type: Literal["boxcox"] = "boxcox"
    param: float = Field(title="Параметр λ преобразования Бокса-Кокса", ge=-1000, le=1000)


# 8. Заполнение пропусков
class FillMissingTransformation(Transformation):
    type: Literal["fillna"] = "fillna"
    method: Literal["last", "backward", "mean", "median", "mode"] = Field(
        title="Метод заполнения пропусков"
    )


# 9. Скользящее среднее
class MovingAverageTransformation(Transformation):
    type: Literal["moving_avg"] = "moving_avg"
    window: int = Field(
        ...,
        gt=0, le=1000,
        title="Размер окна скользящего среднего",
    )


TransformationUnion = Annotated[
    Union[
        DiffTransformation,
        LagTransformation,
        LogTransformation,
        PowTransformation,
        MinMaxTransformation,
        StandardTransformation,
        ExpSmoothTransformation,
        BoxCoxTransformation,
        FillMissingTransformation,
        MovingAverageTransformation
    ],
    Field(discriminator="type")
]


class PreprocessingRequest(BaseModel):
    ts: Timeseries = Field(
        default=Timeseries(name="Временной ряд для преобразования"),
        title="Временной ряд для преобразования"
    )
    transformations: List[TransformationUnion] = Field(title="Список преобразований")


class ContextBase(BaseModel):
    """Контекст предобработки ряда, необходимый для возврата к ряда исходному"""
    step: int = 1


class DiffContext(ContextBase):
    """Контекст для разностей"""
    first_values: list[float] = Field(title='Первые periods значений исходного ряда')

    @model_validator(mode="after")
    def validate_value(self):
        self.first_values = [validate_float_param(value) for value in self.first_values]
        return self


class MinMaxContext(ContextBase):
    """Контекст для нормализации"""
    init_min: float
    init_max: float

    @model_validator(mode="after")
    def validate_value(self):
        self.init_min = validate_float_param(self.init_min)
        self.init_max = validate_float_param(self.init_max)

        return self


class StandardContext(ContextBase):
    """Контекст для стандартизации"""
    init_std: float
    init_mean: float

    @model_validator(mode="after")
    def validate_value(self):
        self.init_std = validate_float_param(self.init_std)
        self.init_mean = validate_float_param(self.init_mean)

        return self


PreprocessContext = Union[
    DiffContext,
    MinMaxContext,
    StandardContext
]


class PreprocessingResponse(BaseModel):
    preprocessed_ts: Timeseries = Field(title='Предобработанный временной ряд')
    contexts: List[PreprocessContext] = Field(title='Список контекстов, необходимых для восстановления ряда')


# Нужны разные схемы т.к. есть предобработки с разными параметрами
# для применения изменений и их отката

class InverseTransformation(BaseModel):
    model_config = ConfigDict(extra="forbid")


class InverseMinMaxTransformation(InverseTransformation):
    type: Literal["minmax"] = "minmax"
    init_min: float = Field(title="Изначальный минимум ряда")
    init_max: float = Field(title="Изначальный максимум ряда")


class InverseStandardTransformation(InverseTransformation):
    type: Literal["standard"] = "standard"
    init_std: float = Field(title="Изначальное стандартное отклонение ряда")
    init_mean: float = Field(title="Изначальное матожидание ряда")


class InverseDiffTransformation(InverseTransformation):
    type: Literal['diff'] = 'diff'
    diff_order: int
    first_values: list[float] = Field(title='Изначальное первое значение ряда')


InverseLagTransformation = LagTransformation
InverseLogTransformation = LogTransformation
InversePowTransformation = PowTransformation
InverseExpSmoothTransformation = ExpSmoothTransformation
InverseBoxCoxTransformation = BoxCoxTransformation
InverseFillMissingTransformation = FillMissingTransformation
InverseMovingAverageTransformation = MovingAverageTransformation

InverseTransformationUnion = Annotated[
    Union[
        InverseDiffTransformation,
        InverseLagTransformation,
        InverseLogTransformation,
        InversePowTransformation,
        InverseMinMaxTransformation,
        InverseStandardTransformation,
        InverseExpSmoothTransformation,
        InverseBoxCoxTransformation,
        InverseFillMissingTransformation,
        InverseMovingAverageTransformation
    ],
    Field(discriminator="type")
]


class InversePreprocessingRequest(BaseModel):
    ts: Timeseries
    transformations: List[InverseTransformationUnion]
