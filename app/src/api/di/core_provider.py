from dishka import Provider, Scope, provide

from src.core.application.building_model.use_cases.fit_arimax import FitArimaxUC
from src.core.application.building_model.use_cases.fit_nhits import FitNhitsUC
from src.core.application.building_model.use_cases.params import ArimaxParamsUC, NhitsParamsUC
from src.core.application.preliminary_diagnosis.use_cases.dicker_fuller import DickeuFullerUC
from src.core.application.preliminary_diagnosis.use_cases.kpss import KpssUC
from src.core.application.preliminary_diagnosis.use_cases.phillips_perron import PhillipsPerronUC


class CoreProvider(Provider):
    scope = Scope.REQUEST

    # команды для построения моделей
    arimax_fit_command = provide(FitArimaxUC, provides=FitArimaxUC)
    nhits_fit_command = provide(FitNhitsUC, provides=FitNhitsUC)

    # параметры моделей
    arimax_params_command = provide(ArimaxParamsUC, provides=ArimaxParamsUC)
    nhits_params_command = provide(NhitsParamsUC, provides=NhitsParamsUC)

    # тесты на стационарность
    dickey_fuller_command = provide(DickeuFullerUC, provides=DickeuFullerUC)
    kpss_command = provide(KpssUC, provides=KpssUC)
    phillips_perron_command = provide(PhillipsPerronUC, provides=PhillipsPerronUC)

