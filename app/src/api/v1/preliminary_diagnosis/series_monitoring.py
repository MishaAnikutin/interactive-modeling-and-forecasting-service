from fastapi import APIRouter
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.use_cases.student_test import StudentTestUC
from src.core.application.preliminary_diagnosis.use_cases.fisher_test import FisherTestUC
from src.core.application.preliminary_diagnosis.use_cases.two_sigma_test import TwoSigmaTestUC

from src.core.application.preliminary_diagnosis.schemas.series_monitoring import (
    StudentTestRequest,
    StudentTestResponse,
    FisherTestRequest,
    FisherTestResponse,
    TwoSigmaTestRequest,
    TwoSigmaTestResponse
)

series_monitoring_router = APIRouter(prefix="/series_monitoring", tags=["Мониторинг и детекция структурных сдвигов у рядов"])


@series_monitoring_router.post(path="/student_test")
@inject_sync
def student_test(
    request: StudentTestRequest,
    uc: FromDishka[StudentTestUC]
) -> StudentTestResponse:
    return uc.execute(request=request)


@series_monitoring_router.post(path="/fisher_test")
@inject_sync
def fisher_test(
    request: FisherTestRequest,
    uc: FromDishka[FisherTestUC]
) -> FisherTestResponse:
    return uc.execute(request=request)


@series_monitoring_router.post(path="/two_sigma_test")
@inject_sync
def two_sigma_test(
    request: TwoSigmaTestRequest,
    uc: FromDishka[TwoSigmaTestUC]
) -> TwoSigmaTestResponse:
    return uc.execute(request=request)
