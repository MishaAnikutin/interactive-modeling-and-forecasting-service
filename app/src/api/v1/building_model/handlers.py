from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject

from .schemas import ArimaxFitResult, ArimaxFitRequest

from src.core.application.building_model.use_cases.fit_arimax import FitArimaxUC


fit_model_router = APIRouter(prefix="/building_model", tags=["Построение модели"])


@fit_model_router.post("/arimax/fit")
@inject
def fit_arimax(
    request: ArimaxFitRequest, fit_arimax_uc: FromDishka[FitArimaxUC]
) -> ArimaxFitResult:
    return fit_arimax_uc.execute(request=request)
