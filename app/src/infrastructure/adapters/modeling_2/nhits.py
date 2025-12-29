from typing import Type

from neuralforecast.models import NHITS

from src.core.application.building_model.schemas.nhits_v2 import NhitsParams_V2
from src.infrastructure.adapters.modeling_2.base import BaseNeuralForecast, TResult


class NhitsAdapter_V2(BaseNeuralForecast[NhitsParams_V2]):
    @property
    def result_class(self) -> Type[TResult]:
        return None

    model_name = "NHITS"
    model_class = NHITS

    def _prepare_model(self, hyperparameters: NhitsParams_V2):
        hyperparameters = hyperparameters.model_dump()
        stack_types = hyperparameters['n_stacks'] * ['identity']
        del hyperparameters['n_stacks']
        h = hyperparameters['output_size']
        del hyperparameters['output_size']
        self.model = self.model_class(
            hist_exog_list=[exog_col for exog_col in self.exog.columns] if self.exog is not None else None,
            accelerator='cpu',
            stack_types=stack_types,
            h=h,
            devices=1,
            **hyperparameters
        )

