from dishka import Provider, Scope, provide

from src.core.application.building_model.use_cases.fit_arimax import FitArimaxUC
from src.core.application.building_model.use_cases.fit_nhits import FitNhitsUC
from src.core.application.building_model.use_cases.params import ArimaxParamsUC, NhitsParamsUC
from src.core.application.preliminary_diagnosis.use_cases.stats_test import StationarityUC


class CoreProvider(Provider):
    scope = Scope.REQUEST

    arimax_fit_command = provide(FitArimaxUC, provides=FitArimaxUC)
    nhits_fit_command = provide(FitNhitsUC, provides=FitNhitsUC)

    arimax_params_command = provide(ArimaxParamsUC, provides=ArimaxParamsUC)
    nhits_params_command = provide(NhitsParamsUC, provides=NhitsParamsUC)

    stationarity_test_command = provide(StationarityUC, provides=StationarityUC)

