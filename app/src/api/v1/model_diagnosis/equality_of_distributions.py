from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.model_diagnosis.schemas.mannwhitney import MannWhitneyRequest, MannWhitneyResponse
from src.core.application.model_diagnosis.schemas.ttest import TtestRequest, TtestResponse
from src.core.application.model_diagnosis.schemas.wilcoxon import WilcoxonRequest, WilcoxonResponse
from src.core.application.model_diagnosis.use_cases.mann_whitney import MannWhitneyUC
from src.core.application.model_diagnosis.use_cases.t_test import TtestUC
from src.core.application.model_diagnosis.use_cases.wilcoxon import WilcoxonUC

equality_of_distributions_router = APIRouter(
    prefix="/equality_of_distributions",
    tags=["Анализ равенства распределений прогнозов"]
)


@equality_of_distributions_router.post('/t_test')
@inject_sync
def t_test(
    request: TtestRequest,
    uc: FromDishka[TtestUC]
) -> TtestResponse:
    return uc.execute(request=request)


@equality_of_distributions_router.post('/mann_whitney')
@inject_sync
def mann_whitney(
    request: MannWhitneyRequest,
    uc: FromDishka[MannWhitneyUC]
) -> MannWhitneyResponse:
    return uc.execute(request=request)


@equality_of_distributions_router.post('/wilcoxon')
@inject_sync
def wilcoxon(
    request: WilcoxonRequest,
    uc: FromDishka[WilcoxonUC]
) -> WilcoxonResponse:
    return uc.execute(request=request)
