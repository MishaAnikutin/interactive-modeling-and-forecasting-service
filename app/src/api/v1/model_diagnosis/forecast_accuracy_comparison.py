
from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.model_diagnosis.schemas.dm_test import DmTestRequest
from src.core.application.model_diagnosis.schemas.mannwhitney import MannWhitneyRequest
from src.core.application.model_diagnosis.schemas.ttest import TtestRequest
from src.core.application.model_diagnosis.schemas.wilcoxon import WilcoxonRequest
from src.core.application.model_diagnosis.use_cases.dm_test import DmTestUC
from src.core.application.model_diagnosis.use_cases.mann_whitney import MannWhitneyUC
from src.core.application.model_diagnosis.use_cases.t_test import TtestUC
from src.core.application.model_diagnosis.use_cases.wilcoxon import WilcoxonUC
from src.core.domain.stat_test.dm_test.result import DmTestResult
from src.core.domain.stat_test.mann_whitney.result import MannWhitneyResult
from src.core.domain.stat_test.ttest.result import TtestResult
from src.core.domain.stat_test.wilcoxon.result import WilcoxonResult


forecast_accuracy_comparison_router = APIRouter(
    prefix="/forecast_accuracy_comparison",
    tags=["Сравнение точности прогнозов"]
)


@forecast_accuracy_comparison_router.post('/dm_test')
@inject_sync
def dm_test(
    request: DmTestRequest,
    uc: FromDishka[DmTestUC]
) -> DmTestResult:
    return uc.execute(request=request)