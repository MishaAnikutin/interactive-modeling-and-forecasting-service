from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.model_diagnosis.schemas.dm_test import DmTestRequest, DmTestResponse
from src.core.application.model_diagnosis.use_cases.dm_test import DmTestUC


forecast_accuracy_comparison_router = APIRouter(
    prefix="/forecast_accuracy_comparison",
    tags=["Сравнение точности прогнозов"]
)


@forecast_accuracy_comparison_router.post('/dm_test')
@inject_sync
def dm_test(
    request: DmTestRequest,
    uc: FromDishka[DmTestUC]
) -> DmTestResponse:
    return uc.execute(request=request)