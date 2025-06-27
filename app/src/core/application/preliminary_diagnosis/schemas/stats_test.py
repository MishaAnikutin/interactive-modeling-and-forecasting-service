from pydantic import BaseModel, Field

from src.core.domain import Timeseries


class StatTestParams(BaseModel):
    test_name: str = Field(title="Название статистического теста", default="DickeyFuller")
    alpha: float = Field(title="Уровень значимости", ge=0, le=1, default=0.05)

class StatTestRequest(BaseModel):
    alpha: float = Field(
        title="Уровень значимости",
        ge=0, le=1,
        default=0.05
    )
    test_names: list[str] = Field(
        title="Список названий тестов для проведения.",
        default=["DickeyFuller"],
    )
    ts: Timeseries

class StatTestResult(StatTestParams):
    stat_value: float = Field(title="Значение статистики теста")
    p_value: float = Field(title="p-value")
