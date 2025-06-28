from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.building_model.schemas.arimax import ArimaxFitRequest, ArimaxFitResult
from src.core.application.building_model.schemas.nhits import NhitsFitRequest, NhitsFitResult
from src.core.application.building_model.use_cases.fit_arimax import FitArimaxUC
from src.core.application.building_model.use_cases.fit_nhits import FitNhitsUC

fit_model_router = APIRouter(prefix="/building_model", tags=["Построение модели"])


@fit_model_router.post("/arimax/fit")
@inject_sync
def fit_arimax(
    request: ArimaxFitRequest,
    fit_arimax_uc: FromDishka[FitArimaxUC]
) -> ArimaxFitResult:
    return fit_arimax_uc.execute(request=request)


@fit_model_router.post("/nhits/fit")
@inject_sync
def fit_nhits(
    request: NhitsFitRequest,
    fit_nhits_uc: FromDishka[FitNhitsUC]
) -> NhitsFitResult:
    return fit_nhits_uc.execute(request=request)
