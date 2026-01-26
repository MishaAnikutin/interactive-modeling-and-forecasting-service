from fastapi import APIRouter, HTTPException
from dishka import FromDishka
from dishka.integrations.fastapi import inject_sync

from src.core.application.preliminary_diagnosis.schemas.break_finder import BreakFinderRequest, BreakFinderResponse
from src.core.application.preliminary_diagnosis.use_cases.break_finder import BreakFinderUC
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
from src.core.domain.stat_test.student import InvalidDateError

series_monitoring_router = APIRouter(prefix="/series_monitoring", tags=["Мониторинг и детекция структурных сдвигов у рядов"])


@series_monitoring_router.post(path="/student_test")
@inject_sync
def student_test(
    request: StudentTestRequest,
    uc: FromDishka[StudentTestUC]
) -> StudentTestResponse:
    try:
        return uc.execute(request=request)
    except InvalidDateError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


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


@series_monitoring_router.post("/break_finder")
@inject_sync
def break_finder(
    request: BreakFinderRequest,
    uc: FromDishka[BreakFinderUC]
) -> BreakFinderResponse:
    try:
        return uc.execute(request=request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=exc)
