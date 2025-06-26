from pydantic import BaseModel, Field

from src.core.domain import Timeseries


class StatTestParams(BaseModel):
    test_name: str = Field(title="Название статистического теста", default="dickey_fuller")
    alpha: float = Field(title="Уровень значимости", ge=0, le=1, default=0.05)

class StatTestRequest(StatTestParams):
    ts: Timeseries

class StatTestResult(StatTestParams):
    stat_value: float = Field(title="Значение статистики теста")
    p_value: float = Field(
        title="p-value",
        description="Вероятность попадания в критическую область при верной нулевой гипотезе"
    )
