from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.errors.kim_andrews import KimAndrewsValidationError
from src.core.application.preliminary_diagnosis.schemas.kim_andrews import KimAndrewsResult, KimAndrewsRequest
from src.core.application.preliminary_diagnosis.use_cases.kim_andrews import KimAndrewsUC

structure_shift_diagnosis_router = APIRouter(prefix="/structure_shift_diagnosis", tags=["Анализ структурного сдвига"])


@structure_shift_diagnosis_router.post(
    path="/kim-andrews",
    responses={
        200: {
            "model": KimAndrewsResult,
            "description": "Успешный ответ",
        },
        422: {
            "model": KimAndrewsValidationError,
            "description": "Ошибка валидации параметров"
        }
    }
)
@inject_sync
def kim_andrews(
    request: KimAndrewsRequest,
    kim_andrews_uc: FromDishka[KimAndrewsUC]
) -> KimAndrewsResult:
    return kim_andrews_uc.execute(request=request)
