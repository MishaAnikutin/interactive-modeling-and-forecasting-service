from dishka import Provider, Scope, provide

from src.core.application.building_model.use_cases.fit_arimax import FitArimaxUC


class CoreProvider(Provider):
    scope = Scope.REQUEST

    arimax_fit_command = provide(FitArimaxUC, provides=FitArimaxUC)
