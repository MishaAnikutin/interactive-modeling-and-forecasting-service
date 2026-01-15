from typing import Optional

from pydantic import BaseModel, Field

from src.core.application.building_model.schemas.arimax import ArimaxFitResult
from src.core.domain import Timeseries, FitParams
from src.core.domain.model.model_data import ModelData
from src.core.domain.parameter_selection.gridsearch_result.arimax import ArimaxGridsearchResult
from src.core.domain.parameter_selection.scoring.information_criteria import InformationCriteriaScoring
from src.core.domain.stat_test.supported_stat_tests import SupportedStationaryTests


class AutoArimaRequest(BaseModel):
    model_data: ModelData
    max_p: int = Field(default=3, description='максимальное значение порядка авторегрессии для перебора')
    max_q: int = Field(default=3, description='максимальное значение порядка скользящего среднего для перебора')
    max_P: int = Field(default=3, description='максимальное значение порядка сезонной авторегрессии для перебора')
    max_D: int = Field(default=1, description='максимальное значение порядка сезонной интеграции для перебора')
    max_Q: int = Field(default=1, description='максимальное значение порядка сезонного скользящего среднего для перебора')
    m: int = Field(default=0, description='Длина сезонного перидоа. Задается вручную')
    stationary_test: SupportedStationaryTests = Field(
        default=SupportedStationaryTests.KPSS,
        description='Какой критерий использовать для подбора параметра d'
    )

    scoring: InformationCriteriaScoring = Field(
        default=InformationCriteriaScoring.aic,
        description='информационный критерий, по которому оптимизируется модель'
    )

    fit_params: FitParams
    # TODO: добавить стратегию кросс-валидации


class AutoArimaResult(BaseModel):
    arimax_fit_result: ArimaxFitResult
    arimax_gridsearch_result: ArimaxGridsearchResult
