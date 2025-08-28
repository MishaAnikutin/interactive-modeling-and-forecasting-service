from typing import Optional, List
from pydantic import BaseModel, Field, model_validator

from src.core.application.model_diagnosis.errors.ljung_box import InvalidLagsError
from src.core.domain import ForecastAnalysis
from src.shared.utils import validate_float_param


class LjungBoxRequest(BaseModel):
    data: ForecastAnalysis = Field(
        title="Прогнозы и исходные данные"
    )

    lags: List[int] | int | None = Field(
        default=None,
        description="Если `lags` — целое число, оно считается максимальным лагом, включенным в тест, "
                    "и результаты теста сообщаются для всех меньших длин лагов. "
                    "Если `lags` — список или массив, включаются все лаги до наибольшего в списке, "
                    "но результаты теста сообщаются только для лагов, указанных в списке. "
                    "Если `lags` равно `None`, то максимальный лаг по умолчанию равен `min(10, nobs // 5)`. "
                    "Количество лагов по умолчанию изменяется, если задан параметр `period`."
    )
    model_df: int = Field(
        default=0,
        ge=0,
        le=1000,
        title="ddof модели",
        description="Число степеней свободы, использованных моделью. В модели ARMA это значение обычно равно p+q, "
                    "где p — порядок авторегрессии (AR), а q — порядок скользящего среднего (MA). "
                    "Это значение вычитается из степеней свободы, используемых в тесте, "
                    "так что скорректированное число степеней свободы для статистики составляет lags - model_df. "
                    "Если lags - model_df <= 0, возвращается NaN."
    )
    period: Optional[int] = Field(
        default=None,
        ge=2,
        le=10000,
        title="Период ряда",
        description="Период сезонного временного ряда. "
                    "Используется для вычисления максимального лага для сезонных данных "
                    "по формуле min(2*период, nobs // 5), если указан. "
                    "Если None, применяется правило по умолчанию для установки количества лагов. "
                    "При задании должен быть >= 2."
    )
    auto_lag: bool = Field(
        default=False,
        title="Флаг оптим. лага",
        description="Флаг, указывающий, следует ли автоматически определять оптимальную "
                    "длину лага на основе порогового значения максимальной корреляции."
    )

    @model_validator(mode="after")
    def validate_lags(self):
        if isinstance(self.lags, int):
            if self.lags <= 0 or self.lags > 10000:
                raise ValueError(InvalidLagsError().detail)
        elif isinstance(self.lags, list):
            for lag in self.lags:
                if lag <= 0 or lag > 10000:
                    raise ValueError(InvalidLagsError().detail)
        return self


class LjungBoxResult(BaseModel):
    lb_stat: List[Optional[float]] = Field(
        title="Статистика теста Льюнга-Бокса",
        description="Значения статистики теста Льюнга-Бокса для каждого указанного лага."
    )

    lb_pvalue: List[Optional[float]] = Field(
        title="P-значения теста Льюнга-Бокса",
        description=(
            "P-значения, рассчитанные на основе хи-квадрат распределения для теста Льюнга-Бокса. "
            "Вычисляется как 1 - chi2.cdf(lb_stat, dof), где dof = lag - model_df. "
            "Если lag - model_df <= 0, возвращается NaN."
        )
    )

    bp_stat: List[Optional[float]] = Field(
        title="Статистика теста Бокса-Пирса",
        description=(
            "Значения статистики теста Бокса-Пирса для каждого указанного лага. "
        )
    )

    bp_pvalue: List[Optional[float]] = Field(
        title="P-значения теста Бокса-Пирса",
        description=(
            "P-значения, рассчитанные на основе хи-квадрат распределения для теста Бокса-Пирса. "
            "Вычисляется как 1 - chi2.cdf(bp_stat, dof), где dof = lag - model_df. "
            "Если lag - model_df <= 0, возвращается NaN. "
        )
    )

    @model_validator(mode="after")
    def validate_floats(self):
        lb_stat = []
        for stat in self.lb_stat:
            lb_stat.append(validate_float_param(stat))

        lb_pvalue = []
        for pvalue in self.lb_pvalue:
            lb_pvalue.append(validate_float_param(pvalue))

        bp_stat = []
        for stat in self.bp_stat:
            bp_stat.append(validate_float_param(stat))

        bp_pvalue = []
        for pvalue in self.bp_pvalue:
            bp_pvalue.append(validate_float_param(pvalue))

        self.lb_stat = lb_stat
        self.lb_pvalue = lb_pvalue
        self.bp_stat = bp_stat
        self.bp_pvalue = bp_pvalue
        return self


