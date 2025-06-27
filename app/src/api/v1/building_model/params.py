from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.building_model.schemas.arimax import ArimaxParams
from src.core.application.building_model.schemas.nhits import NhitsParams
from src.core.application.building_model.use_cases.params import ArimaxParamsUC, NhitsParamsUC

params_router = APIRouter(prefix="/params", tags=["Параметры модели"])


@params_router.get("/arimax")
@inject_sync
def params_arimax(
    arimax_params_uc: FromDishka[ArimaxParamsUC]
) -> ArimaxParams:
    return arimax_params_uc.execute()


@params_router.get("/nhits")
@inject_sync
def params_nhits(
    nhits_params_uc: FromDishka[NhitsParamsUC]
) -> NhitsParams:
    return nhits_params_uc.execute()