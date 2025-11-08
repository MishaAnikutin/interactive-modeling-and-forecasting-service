from dishka import FromDishka
from fastapi import APIRouter, HTTPException
from dishka.integrations.fastapi import inject_sync

from src.core.application.validate_series import (
    ValidationRequest,
    ValidationResponse
)
from src.core.application.validate_series import ValidateSeriesUC

validation_router = APIRouter(prefix="/validate_series", tags=["Валидация рядов"])


@validation_router.post("/", summary='Проверить ряд')
@inject_sync
def preprocessing(
    request: ValidationRequest,
    validate_uc: FromDishka[ValidateSeriesUC],
) -> ValidationResponse:
    try:
        return validate_uc.execute(request)
    except ValueError as detail:
        raise HTTPException(status_code=400, detail=str(detail))
