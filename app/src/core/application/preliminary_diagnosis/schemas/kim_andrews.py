from pydantic import BaseModel, Field, model_validator

from src.core.application.preliminary_diagnosis.schemas.common import StatTestParams, ResultValues


class KimAndrewsRequest(StatTestParams):
    n: int = Field(default=1, title="Размер выборки до сдвига", ge=1, le=100000)
    m: int = Field(default=1, title="Длина хвоста выборки (после сдвига), Разумные значения не превышают 10.", ge=1, le=100)
    shift: int = Field(default=1, title="Параметр сдвига ряда", ge=1, le=100000)
    trend: bool = Field(default=False, title="Добавить трендовую компоненту в OLS")
    const: bool = Field(default=True, title="Добавить константу в OLS")

    @model_validator(mode='after')
    def validate_params(self):
        ts_len = len(self.ts.values)
        na_cnt = sum([(v is None) for v in self.ts.values])
        if self.shift >= ts_len:
            raise ValueError("shift должен быть меньше длины ряда")

        t_eff = ts_len - self.shift - na_cnt

        if self.n + self.m > t_eff:
            raise ValueError(
                f"После сдвига остаётся только {t_eff} наблюдений, "
                f"а требуется минимум n + m = {self.n + self.m}"
            )

        if self.n <= self.m + 1:
            raise ValueError("Необходимо n > m + 1")

        return self


class KimAndrewsResult(BaseModel):
    """Коллекция статистик и их p-value для теста Ким-Эндрюса"""
    Sa_values: ResultValues
    Sb_values: ResultValues
    Sc_values: ResultValues
    Sd_values: ResultValues
    R_values: ResultValues
