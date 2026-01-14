from pydantic import Field, model_validator

from src.core.application.building_model.schemas import NhitsFitRequest
from src.core.application.building_model.schemas.nhits import NhitsParams
from src.core.domain import ForecastResult_V2


class NhitsParams_V2(NhitsParams):
    input_size: int = Field(
        ge=-1, le=1000, default=8, title='Размер входного слоя',
        description='Если поставить -1, то автоматически будет выставлено 3 * output_size, '
                    'также недопустимо значение равное 0'
    )
    output_size: int = Field(ge=1, le=1000, default=2, title='Размер выходного слоя')

    @model_validator(mode='after')
    def validate_input_size(self):
        if self.input_size == 0:
            raise ValueError("input_size должен быть > 0")
        return self


class NhitsFitRequest_V2(NhitsFitRequest):
    hyperparameters: NhitsParams_V2 = Field(title="Параметры модели NHiTS")


class NhitsFitResult_V2(ForecastResult_V2):
    pass
