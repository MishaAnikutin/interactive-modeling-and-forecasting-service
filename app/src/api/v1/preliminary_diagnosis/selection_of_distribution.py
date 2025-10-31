from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync
from src.core.application.preliminary_diagnosis.schemas.select_distribution import SelectDistRequest, SelectDistResponse
from src.core.application.preliminary_diagnosis.use_cases.select_distribution import SelectDistUC

selection_of_distribution_router = APIRouter(prefix="/distribution", tags=["Подбор распределения"])


@selection_of_distribution_router.post(path="/find")
@inject_sync
def find(
    request: SelectDistRequest,
    select_dist_uc: FromDishka[SelectDistUC]
) -> SelectDistResponse:
    return select_dist_uc.execute(request=request)
