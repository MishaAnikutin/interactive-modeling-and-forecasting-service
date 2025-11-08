from typing import List, Optional

from pydantic import Field, BaseModel

from src.core.domain import Timeseries
from src.core.domain.validation import ValidationIssue


class ValidationRequest(BaseModel):
    ts: Timeseries


class ValidationResponse(BaseModel):
    issues: List[ValidationIssue] = Field(..., title="Список ошибок валидаций. Если пустой значит все хорошо")
