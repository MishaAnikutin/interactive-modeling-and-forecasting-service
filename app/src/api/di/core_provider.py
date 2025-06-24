from dishka import Provider, Scope, provide

from src.core.application.building_model.use_cases.fit_arimax import FitArimaxUC
from src.core.application.building_model.use_cases.params import ArimaxParamsUC


class CoreProvider(Provider):
    scope = Scope.REQUEST

    arimax_fit_command = provide(FitArimaxUC, provides=FitArimaxUC)
    arimax_params_command = provide(ArimaxParamsUC, provides=ArimaxParamsUC)
