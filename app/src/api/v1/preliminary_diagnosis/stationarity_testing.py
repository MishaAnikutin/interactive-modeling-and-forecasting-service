from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.dickey_fuller import DickeyFullerParams, DickeyFullerResult
from src.core.application.preliminary_diagnosis.use_cases.dicker_fuller import DickeuFullerUC

stationary_testing_router = APIRouter(prefix="/stationary_testing", tags=["Анализ ряда на стационарность"])

@stationary_testing_router.get("/dickey_fuller")
@inject_sync
def dickey_fuller(
    request: DickeyFullerParams,
    dickey_fuller_uc: FromDishka[DickeuFullerUC]
) -> DickeyFullerResult:
    return dickey_fuller_uc.execute(request=request)


