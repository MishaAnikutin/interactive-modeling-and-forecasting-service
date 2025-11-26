from typing import Annotated, Union, Literal

from pydantic import BaseModel, Field

from src.core.application.building_model.errors.alignment import NotEqualToExpectedError, NotConstantFreqError, \
    NoDataAfterAlignmentError, NotLastDayOfMonthError, NotSupportedFreqError, EmptyError, BoundariesError


class ListLengthError(BaseModel):
    type: Literal["length"] = "length"
    detail: str = Field(
        default="Все списки должны иметь одинаковую длину: n_stacks = len(n_blocks) = len(n_pool_kernel_size)",
        title="Описание ошибки"
    )


class KernelSizeError(BaseModel):
    type: Literal["kernel"] = "kernel"
    detail: str = Field(
        default="Все значения в списке n_pool_kernel_size должны быть больше или равны 1",
        title="Описание ошибки"
    )


PydanticValidationErrorType = Annotated[
    Union[
        ListLengthError,
        KernelSizeError,
        BoundariesError,
    ],
    Field(discriminator="type")
]


class NhitsPydanticValidationError(BaseModel):
    msg: PydanticValidationErrorType = Field(
        title="Описание ошибки",
        default=ListLengthError()
    )


class HorizonValidationError(BaseModel):
    type: Literal["horizon"] = "horizon"
    detail: str = Field(
        default="Горизонт прогноза + размер тестовой выборки должен быть больше 0",
        title="Описание ошибки"
    )


class ValSizeError(BaseModel):
    type: Literal["val_size"] = "val_size"
    detail: str = Field(
        default=f"Недостаточный объем валидационной выборки для заданной схемы разбиения временного ряда. "
                f"При текущих параметрах: val_size=, test_size=, h= (горизонт прогноза). "
                f"Минимально допустимый размер валидационной выборки должен удовлетворять условию: val_size >= (h + test_size) "
                f"или val_size = 0",
        title="Описание ошибки валидационной выборки"
    )

    def __init__(self, val_size: int, test_size: int, h: int, **data):
        if val_size == 0:
            detail_msg = "Размер валидационной выборки не может быть равен 0 при данных параметрах"
        else:
            detail_msg = (
                f"Недостаточный объем валидационной выборки для заданной схемы разбиения временного ряда. "
                f"При текущих параметрах: val_size={val_size}, test_size={test_size}, h={h - test_size} (горизонт прогноза). "
                f"Минимально допустимый размер валидационной выборки должен удовлетворять условию: val_size >= {h} (h + test_size) "
                f"или val_size = 0."
            )

        super().__init__(detail=detail_msg, **data)


class PatienceStepsError(BaseModel):
    type: Literal["patience"] = "patience"
    detail: str = Field(
        default=(
            "Валидационная выборка должна быть не пустой, "
            "если ранняя остановка включена (early_stop_patience_steps > 0)"
        ),
        title="Описание ошибки"
    )


class TrainSizeError(BaseModel):
    type: Literal["train"] = "train_size"
    detail: str = Field(
        default=(
            "4 * (h + размер тестовой выборки) должно быть <= размер обучающей выборки"
        ),
        title="Описание ошибки"
    )


FitValidationErrorType = Annotated[
    Union[
        HorizonValidationError,
        ValSizeError,
        PatienceStepsError,
        TrainSizeError,
        NotEqualToExpectedError,
        NoDataAfterAlignmentError,
        NotConstantFreqError,
        NotSupportedFreqError,
        NotLastDayOfMonthError,
        EmptyError
    ],
    Field(discriminator="type")
]

class NhitsFitValidationError(BaseModel):
    msg: FitValidationErrorType = Field(
        title="Описание ошибки",
        default=HorizonValidationError()
    )