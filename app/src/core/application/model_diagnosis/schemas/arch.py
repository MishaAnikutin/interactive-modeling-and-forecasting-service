from typing import Optional, Literal

from pydantic import BaseModel, Field

from src.core.application.model_diagnosis.schemas.common import ResidAnalysisData


class ArchRequest(BaseModel):
    data: ResidAnalysisData = Field(
        title="Прогноз и исхдоные данные"
    )
    nlags: Optional[int] = Field(
        default=None,
        ge=0, le=10000,
        title="Максимальный лаг"
    )

    period: Optional[int] = Field(
        default=None,
        title="Период",
        ge=2, le=10000,
        description=(
            "Период сезонного временного ряда. "
            "Используется для вычисления максимального лага для сезонных "
            "данных по формуле min(2*период, nobs // 5), если указан. "
            "Если None, применяется правило по умолчанию для установки количества лагов. "
            "При задании должен быть >= 2."
        )
    )

    ddof: int = Field(
        default=0,
        title="Число степеней свободы",
        ge=0,
        le=1000
    )

    cov_type: Literal[
        "nonrobust", "fixed scale",
        "HC0", "HC1", "HC2",
        "HC3", "HAC", "hac-panel",
        "hac-groupsum", "cluster"
    ] = Field(
        default="nonrobust",
        title="Тип ковариации",
        description="Значение по умолчанию - nonrobust, который использует классическую оценку ковариации OLS. "
                    "Укажите одно из значений “HC0”, “HC1”, “HC2”, “HC3”, чтобы использовать оценку ковариации Уайта. "
                    "Принимаются все типы ковариаций, поддерживаемые OLS.fit."
    )

