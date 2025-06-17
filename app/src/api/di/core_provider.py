from dishka import Provider, Scope, provide

from src.core import FitArimaxUC


class CoreProvider(Provider):
    scope = Scope.REQUEST

    arimax_fit_command = provide(FitArimaxUC, provides=FitArimaxUC)
