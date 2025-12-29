from pydantic import Field, model_validator

from src.core.application.building_model.schemas.lstm import LstmParams, LstmFitRequest
from src.core.domain import ForecastResult_V2


class LstmParams_V2(LstmParams):
    input_size: int = Field(
        ge=-1, default=8, title='Размер входного слоя',
        description='Если поставить -1, то автоматически будет выставлено 3 * output_size, '
                    'также недопустимо значение равное 0'
    )
    output_size: int = Field(ge=1, default=2, title='Размер выходного слоя')

    @model_validator(mode='after')
    def validate_input_size(self):
        if self.val_size == 0:
            raise ValueError("input_size должен быть > 0")
        return self


class LstmFitRequest_V2(LstmFitRequest):
    hyperparameters: LstmParams_V2 = Field(title="Параметры модели NHiTS")


class LstmFitResult_V2(ForecastResult_V2):
    pass
