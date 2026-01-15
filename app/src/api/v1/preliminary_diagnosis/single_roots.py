from fastapi import APIRouter, HTTPException
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.hegy import HegyRequest, HegyResponse
from src.core.application.preliminary_diagnosis.use_cases.hegy import HegyUC

single_roots_router = APIRouter(prefix="/single_roots", tags=["Тесты на единичные корни"])


@single_roots_router.post(path="/hegy")
@inject_sync
def hegy(
    request: HegyRequest,
    uc: FromDishka[HegyUC]
) -> HegyResponse:
    try:
        return uc.execute(request=request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
