from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.cv_value import VariationCoeffParams, VariationCoeffResult
from src.core.application.preliminary_diagnosis.schemas.kurtosis import KurtosisParams, KurtosisResult
from src.core.application.preliminary_diagnosis.schemas.mean_value import MeanResult, MeanParams
from src.core.application.preliminary_diagnosis.schemas.median_value import MedianParams, MedianResult
from src.core.application.preliminary_diagnosis.schemas.mode_value import ModeParams, ModeResult
from src.core.application.preliminary_diagnosis.schemas.quantiles import QuantilesResult, QuantilesParams
from src.core.application.preliminary_diagnosis.schemas.skewness import SkewnessParams, SkewnessResult
from src.core.application.preliminary_diagnosis.schemas.var_value import VarianceParams, VarianceResult
from src.core.application.preliminary_diagnosis.use_cases.cv_value import VariationCoeffUC
from src.core.application.preliminary_diagnosis.use_cases.kurtosis import KurtosisUC
from src.core.application.preliminary_diagnosis.use_cases.mean_value import MeanUC
from src.core.application.preliminary_diagnosis.use_cases.median_value import MedianUC
from src.core.application.preliminary_diagnosis.use_cases.mode_value import ModeUC
from src.core.application.preliminary_diagnosis.use_cases.quantiles import QuantilesUC
from src.core.application.preliminary_diagnosis.use_cases.skewness import SkewnessUC
from src.core.application.preliminary_diagnosis.use_cases.var_value import VarianceUC

descriptive_statistics_router = APIRouter(prefix="/descriptive_statistics", tags=["Описательная статистика"])


@descriptive_statistics_router.post(path="/mean")
@inject_sync
def mean_value(
    request: MeanParams,
    mean_uc: FromDishka[MeanUC]
) -> MeanResult:
    return mean_uc.execute(request=request)

@descriptive_statistics_router.post(path="/median")
@inject_sync
def median_value(
    request: MedianParams,
    median_uc: FromDishka[MedianUC]
) -> MedianResult:
    return median_uc.execute(request=request)

@descriptive_statistics_router.post(path="/mode")
@inject_sync
def mode_value(
    request: ModeParams,
    mode_uc: FromDishka[ModeUC]
) -> ModeResult:
    return mode_uc.execute(request=request)

@descriptive_statistics_router.post(path="/variance")
@inject_sync
def variance_value(
    request: VarianceParams,
    var_uc: FromDishka[VarianceUC]
) -> VarianceResult:
    return var_uc.execute(request=request)

@descriptive_statistics_router.post(path="/coff_of_variation")
@inject_sync
def cv_value(
    request: VariationCoeffParams,
    cv_uc: FromDishka[VariationCoeffUC]
) -> VariationCoeffResult:
    return cv_uc.execute(request=request)


@descriptive_statistics_router.post(path="/quantiles")
@inject_sync
def quantiles(
    request: QuantilesParams,
    quant_uc: FromDishka[QuantilesUC]
) -> QuantilesResult:
    return quant_uc.execute(request=request)

@descriptive_statistics_router.post(path="/skewness")
@inject_sync
def skewness(
    request: SkewnessParams,
    skewness_uc: FromDishka[SkewnessUC]
) -> SkewnessResult:
    return skewness_uc.execute(request=request)

@descriptive_statistics_router.post(path="/kurtosis")
@inject_sync
def kurtosis(
    request: KurtosisParams,
    kurtosis_uc: FromDishka[KurtosisUC]
) -> KurtosisResult:
    return kurtosis_uc.execute(request=request)