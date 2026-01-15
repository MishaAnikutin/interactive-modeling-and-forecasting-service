from pydantic import BaseModel, Field

from src.core.application.building_model.schemas.arimax import ArimaxParams


class ArimaxGridsearchResult(BaseModel):
    optimal_params: ArimaxParams  = Field(..., title='Подобранные параметры для ARIMA')
    information_criteria_value: float = Field(..., title='Значение информационного критерия')
    short_representation: str = Field(default='ARIMA(0,0,1)(2,0,1)[4]', title='Краткое представление модели')
