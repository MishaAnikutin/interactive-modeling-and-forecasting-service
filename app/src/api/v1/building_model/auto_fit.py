from fastapi import APIRouter, Response, HTTPException
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.building_model.schemas.autoarima import AutoArimaRequest
from src.core.application.building_model.use_cases.autoarima import AutoArimaUC
from src.infrastructure.adapters.modeling.errors.arimax import ConstantInExogAndSpecification

automl_model_router = APIRouter(prefix="/building_model/auto", tags=["Модели с автоподбором параметров"])


@automl_model_router.post("/autoarimax/fit")
@inject_sync
def autoarimax(
    request: AutoArimaRequest,
    autoarimax_uc: FromDishka[AutoArimaUC]
) -> Response:
    try:
        archive_response = autoarimax_uc.execute(request=request)
    except ConstantInExogAndSpecification as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return Response(
        content=archive_response,
        media_type="application/octet-stream",
    )
