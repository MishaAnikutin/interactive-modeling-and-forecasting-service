from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.generating_series.schemas.naive_decomposition import NaiveDecompositionRequest, \
    NaiveDecompositionResult
from src.core.application.generating_series.schemas.stl_decomposition import STLDecompositionRequest, \
    STLDecompositionResult
from src.core.application.generating_series.use_cases.naive_decomposition import NaiveDecompositionUC
from src.core.application.generating_series.use_cases.stl_decomposition import STLDecompositionUC

seasonal_decomposition_router = APIRouter(
    prefix="/seasonal_decomposition",
    tags=["Сезонная декомпозиция"]
)

@seasonal_decomposition_router.post("/stl")
@inject_sync
def stl_decomposition(
    request: STLDecompositionRequest,
    stl_decomposition_uc: FromDishka[STLDecompositionUC]
) -> STLDecompositionResult:
    return stl_decomposition_uc.execute(request=request)


@seasonal_decomposition_router.post("/naive")
@inject_sync
def naive_decomposition(
    request: NaiveDecompositionRequest,
    naive_decomposition_uc: FromDishka[NaiveDecompositionUC]
) -> NaiveDecompositionResult:
    return naive_decomposition_uc.execute(request=request)

