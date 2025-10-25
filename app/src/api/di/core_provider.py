from dishka import Provider, Scope, provide

from src.core.application.building_model.use_cases.models import FitArimaxUC, FitGruUC, FitLstmUC, FitNhitsUC
from src.core.application.predict_series.use_cases.predict_arimax import PredictArimaxUC
from src.core.application.generating_series.use_cases.naive_decomposition import NaiveDecompositionUC
from src.core.application.generating_series.use_cases.stl_decomposition import STLDecompositionUC
from src.core.application.model_diagnosis.use_cases.arch import ArchUC
from src.core.application.model_diagnosis.use_cases.breusch_godfrey import AcorrBreuschGodfreyUC
from src.core.application.model_diagnosis.use_cases.jarque_bera import JarqueBeraUC
from src.core.application.model_diagnosis.use_cases.kstest import KolmogorovUC
from src.core.application.model_diagnosis.use_cases.ljung_box import LjungBoxUC
from src.core.application.model_diagnosis.use_cases.lm import LmUC
from src.core.application.model_diagnosis.use_cases.omnibus import OmnibusUC
from src.core.application.predict_series.use_cases.predict_gru import PredictGruUC
from src.core.application.predict_series.use_cases.predict_lstm import PredictLstmUC
from src.core.application.predict_series.use_cases.predict_nhits import PredictNhitsUC
from src.core.application.preliminary_diagnosis.use_cases.cv_value import VariationCoeffUC
from src.core.application.preliminary_diagnosis.use_cases.df_gls import DfGlsUC
from src.core.application.preliminary_diagnosis.use_cases.dicker_fuller import DickeuFullerUC
from src.core.application.preliminary_diagnosis.use_cases.kde import KdeUC
from src.core.application.preliminary_diagnosis.use_cases.kpss import KpssUC
from src.core.application.preliminary_diagnosis.use_cases.kurtosis import KurtosisUC
from src.core.application.preliminary_diagnosis.use_cases.mean_value import MeanUC
from src.core.application.preliminary_diagnosis.use_cases.median_value import MedianUC
from src.core.application.preliminary_diagnosis.use_cases.mode_value import ModeUC
from src.core.application.preliminary_diagnosis.use_cases.phillips_perron import PhillipsPerronUC
from src.core.application.preliminary_diagnosis.use_cases.pp_plot import PPplotUC
from src.core.application.preliminary_diagnosis.use_cases.qq import QQplotUC
from src.core.application.preliminary_diagnosis.use_cases.quantiles import QuantilesUC
from src.core.application.preliminary_diagnosis.use_cases.range_uc import RangeUnitRootUC
from src.core.application.preliminary_diagnosis.use_cases.skewness import SkewnessUC
from src.core.application.preliminary_diagnosis.use_cases.var_value import VarianceUC
from src.core.application.preliminary_diagnosis.use_cases.zivot_andrews import ZivotAndrewsUC
from src.core.application.preprocessing.preprocessing_uc import PreprocessUC
from src.core.application.preprocessing.inverse_preprocessing_uc import InversePreprocessUC
from src.core.application.preliminary_diagnosis.use_cases.acf_and_pacf import AcfAndPacfUC


# TODO: возмонжо стоит разделить провайдеров на каждый use case ...
class CoreProvider(Provider):
    scope = Scope.REQUEST

    # команды для построения моделей
    arimax_fit_command = provide(FitArimaxUC, provides=FitArimaxUC)
    nhits_fit_command = provide(FitNhitsUC, provides=FitNhitsUC)
    lstm_fit_command = provide(FitLstmUC, provides=FitLstmUC)
    gru_fit_command = provide(FitGruUC, provides=FitGruUC)

    # команды для прогнозирования моделями
    arimax_predict_command = provide(PredictArimaxUC, provides=PredictArimaxUC)
    gru_predict_command = provide(PredictGruUC, provides=PredictGruUC)
    lstm_predict_command = provide(PredictLstmUC, provides=PredictLstmUC)
    nhits_predict_command = provide(PredictNhitsUC, provides=PredictNhitsUC)

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
    lm_command = provide(LmUC, provides=LmUC)

    # Анализ автокорреляции
    ljung_box_command = provide(LjungBoxUC, provides=LjungBoxUC)
    breusch_godfrey_command = provide(AcorrBreuschGodfreyUC, provides=AcorrBreuschGodfreyUC)
    acf_pacf_command = provide(AcfAndPacfUC, provides=AcfAndPacfUC)

    # Команды для описательной статистики
    mean_command = provide(MeanUC, provides=MeanUC)
    median_command = provide(MedianUC, provides=MedianUC)
    mode_command = provide(ModeUC, provides=ModeUC)
    variance_command = provide(VarianceUC, provides=VarianceUC)
    cv_command = provide(VariationCoeffUC, provides=VariationCoeffUC)
    quantiles_command = provide(QuantilesUC, provides=QuantilesUC)
    skewness_command = provide(SkewnessUC, provides=SkewnessUC)
    kurtosis_command = provide(KurtosisUC, provides=KurtosisUC)

    # Распределения
    qq_plot_command = provide(QQplotUC, provides=QQplotUC)
    pp_plot_command = provide(PPplotUC, provides=PPplotUC)
    kde_command = provide(KdeUC, provides=KdeUC)
