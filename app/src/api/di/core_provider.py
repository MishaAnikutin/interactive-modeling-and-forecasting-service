from dishka import Provider, Scope, provide

from src.core.application.building_model.use_cases.fit_arimax import FitArimaxUC
from src.core.application.building_model.use_cases.fit_gru import FitGruUC
from src.core.application.building_model.use_cases.fit_lstm import FitLstmUC
from src.core.application.building_model.use_cases.fit_nhits import FitNhitsUC
from src.core.application.building_model.use_cases.params import ArimaxParamsUC, NhitsParamsUC
from src.core.application.generating_series.use_cases.naive_decomposition import NaiveDecompositionUC
from src.core.application.generating_series.use_cases.stl_decomposition import STLDecompositionUC
from src.core.application.model_diagnosis.use_cases.arch import ArchUC
from src.core.application.model_diagnosis.use_cases.breusch_godfrey import AcorrBreuschGodfreyUC
from src.core.application.model_diagnosis.use_cases.jarque_bera import JarqueBeraUC
from src.core.application.model_diagnosis.use_cases.kstest import KolmogorovUC
from src.core.application.model_diagnosis.use_cases.ljung_box import LjungBoxUC
from src.core.application.model_diagnosis.use_cases.lm import LmUC
from src.core.application.model_diagnosis.use_cases.omnibus import OmnibusUC
from src.core.application.preliminary_diagnosis.use_cases.df_gls import DfGlsUC
from src.core.application.preliminary_diagnosis.use_cases.dicker_fuller import DickeuFullerUC
from src.core.application.preliminary_diagnosis.use_cases.kpss import KpssUC
from src.core.application.preliminary_diagnosis.use_cases.phillips_perron import PhillipsPerronUC
from src.core.application.preliminary_diagnosis.use_cases.range_uc import RangeUnitRootUC
from src.core.application.preliminary_diagnosis.use_cases.zivot_andrews import ZivotAndrewsUC
from src.core.application.preprocessing.preprocessing_uc import PreprocessUC
from src.core.application.preprocessing.inverse_preprocessing_uc import InversePreprocessUC


class CoreProvider(Provider):
    scope = Scope.REQUEST

    # команды для построения моделей
    arimax_fit_command = provide(FitArimaxUC, provides=FitArimaxUC)
    nhits_fit_command = provide(FitNhitsUC, provides=FitNhitsUC)
    lstm_fit_command = provide(FitLstmUC, provides=FitLstmUC)
    gru_fit_command = provide(FitGruUC, provides=FitGruUC)

    # параметры моделей
    arimax_params_command = provide(ArimaxParamsUC, provides=ArimaxParamsUC)
    nhits_params_command = provide(NhitsParamsUC, provides=NhitsParamsUC)

    # тесты на стационарность
    dickey_fuller_command = provide(DickeuFullerUC, provides=DickeuFullerUC)
    kpss_command = provide(KpssUC, provides=KpssUC)
    phillips_perron_command = provide(PhillipsPerronUC, provides=PhillipsPerronUC)
    df_gls_command = provide(DfGlsUC, provides=DfGlsUC)
    zivot_andrews_command = provide(ZivotAndrewsUC, provides=ZivotAndrewsUC)
    range_unit_root_command = provide(RangeUnitRootUC, provides=RangeUnitRootUC)

    # предобработка ряда
    preprocess_command = provide(PreprocessUC, provides=PreprocessUC)
    undo_preprocess_command = provide(InversePreprocessUC, provides=InversePreprocessUC)

    # команды для разложения ряда
    stl_decomposition_command = provide(STLDecompositionUC, provides=STLDecompositionUC)
    naive_decomposition_command = provide(NaiveDecompositionUC, provides=NaiveDecompositionUC)

    # команды для анализа остатков
    omnibus_command = provide(OmnibusUC, provides=OmnibusUC)
    jarque_bera_command = provide(JarqueBeraUC, provides=JarqueBeraUC)
    kstest_command = provide(KolmogorovUC, provides=KolmogorovUC)
    arch_command = provide(ArchUC, provides=ArchUC)
    breusch_godfrey_command = provide(AcorrBreuschGodfreyUC, provides=AcorrBreuschGodfreyUC)
    lm_command = provide(LmUC, provides=LmUC)
    ljung_box_command = provide(LjungBoxUC, provides=LjungBoxUC)
