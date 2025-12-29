from typing import Literal

from pydantic import BaseModel, Field

detail_msg_1 = (
    "Недостаточный объем валидационной выборки для заданной схемы разбиения временного ряда. "
    "При текущих параметрах: val_size=%d, output_size=%d. "
    "Минимально допустимый размер валидационной выборки должен удовлетворять условию: "
    "val_size >= output_size или валидационная выборка должна быть пустой."
)

detail_msg_2 = (
    "Недостаточный размер обучающей выборки. "
    "Получено: train_size=%d, input_size=%d, output_size=%d. "
    "Требуется: train_size >= %d = (input_size + output_size)"
)

detail_msg_3 = (
    "Недостаточный размер обучающей выборки для LSTM / GRU модели. "
    "Получено: train_size=%d, input_size=%d, h_train=%d. "
    "Требуется: train_size >= %d = (input_size + h_train)"
)

detail_msg_4 = (
    "Валидационная выборка должна быть не пустой, "
    "если ранняя остановка включена (early_stop_patience_steps > 0)"
)

class ValSizeError(BaseModel):
    type: Literal["val_size"] = "val_size"
    detail: str = Field(
        default=detail_msg_1,
        title="Описание ошибки валидационной выборки"
    )

    def __init__(self, val_size: int, output_size: int, **data):
        detail = detail_msg_1 % (val_size, output_size)
        super().__init__(detail=detail, **data)


class PatienceStepsError(BaseModel):
    type: Literal["patience"] = "patience"
    detail: str = Field(
        default=detail_msg_4,
        title="Описание ошибки"
    )

class TrainSizeError(BaseModel):
    type: Literal["train_size"] = "train_size"
    detail: str = Field(
        default=detail_msg_2,
        title="Описание ошибки обучающей выборки"
    )

    def __init__(self, train_size: int, input_size: int, output_size: int, **data):
        required_min = input_size + output_size
        detail = detail_msg_2 % (train_size, input_size, output_size, required_min)
        super().__init__(detail=detail, **data)


class LSTM_GRU_TrainSizeError(BaseModel):
    type: Literal["train_size_2"] = "train_size_2"
    detail: str = Field(
        default=detail_msg_3,
        title="Описание ошибки обучающей выборки"
    )

    def __init__(self, train_size: int, input_size: int, h_train: int, **data):
        required_min = input_size + h_train
        detail = detail_msg_3 % (train_size, input_size, h_train, required_min)
        super().__init__(detail=detail, **data)