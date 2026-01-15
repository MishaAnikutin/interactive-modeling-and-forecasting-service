from fastapi import APIRouter, HTTPException
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.spurious_regression import SpuriousRegressionRequest, \
    SpuriousRegressionResponse
from src.core.application.preliminary_diagnosis.use_cases.spurious_regression import SpuriousRegressionUC

spurious_regression_router = APIRouter(prefix="/spurious_regression", tags=["Тесты на мнимую регрессию"])


@spurious_regression_router.post(path="/check")
@inject_sync
def check(
    request: SpuriousRegressionRequest,
    uc: FromDishka[SpuriousRegressionUC]
) -> SpuriousRegressionResponse:
    return uc.execute(request=request)
