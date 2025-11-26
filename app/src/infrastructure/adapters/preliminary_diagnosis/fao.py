from typing import Optional

import pandas as pd
import pandas.core.indexes.datetimes as pdi
import pandas.api.types as pda
import numpy as np
from fastapi import HTTPException

from src.core.application.preliminary_diagnosis.errors.fao import SmallSizeError
from src.core.application.preliminary_diagnosis.schemas.fao import FaoResult, FaoEnum


def ipa_ranges(x):
    """
    Данная функция переводит IPA из численного вида в понятный пользователям.
    """
    if x <= -1:
        return "Alert low"
    elif x <= -0.5:
        return "Watch low"
    elif x >= 1:
        return "Alert high"
    elif x >= 0.5:
        return "Watch high"
    elif np.isnan(x):
        return ""
    return "Normal"


def proc_res(x):
    """
    Данная функция переводит IPA из понятнного пользователям вида в числовое значение {0,1,2}.
    Нужна для построения графика цен с выделением периодов, требующих внимания.
    """
    match x:
        case "Watch low" | "Watch high":
            return 1
        case "Alert low" | "Alert high":
            return 2
        case _:
            return 0


def get_ipa(
    dataset: pd.DataFrame,
    price: str | None = None,
    date: str | None = None,
    quarter=False,
) -> pd.DataFrame:
    """
    Данная функция рассчитывает Индикатор Ценовых Аномалий (IPA).
    IPA рассчитывается на основе месячных данных и выдает результат с годовой или квартальной частотой.

    Parameters
    ----------
    dataset : pd.DataFrame
        Источник данных для расчета IPA.

    price : str
        Имя столбца в данных, содержащего данные по ценам.

    date : str | None
        Имя столбца, содержащего дату каждого наблюдения.
        Дата должна быть в формате, содержащем атрибуты `year` и `month`.

        Если имя переменной не задано, то функция пытается использовать индекс датасета.
        В этом случае индекс должен быть типа `DatetimeIndex`.

    quarter : bool
        Если `True`, то требуется рассчитать квартальный IPA.
        В противном случае рассчитывается годовой IPA.

    Returns
    _______
    pandas.DataFrame
        Датафрейм, содержащий единственный столбец со значениями IPA.

    Raises
    ------
    AssertionError
        Что-то не то со входными данными.
    """
    assert price is not None, "No price data provided!"
    if date is not None:
        assert date in dataset, f"No `{date}` in dataframe!"
        assert pda.is_datetime64_dtype(dataset[date]), f"`{date}` is not datetime64!"
        dt = dataset.loc[:, [date, price]]
        dt.set_index(date, inplace=True)
    else:
        assert (
            type(dataset.index) == pdi.DatetimeIndex
        ), "The index of the `dataset` should be Datetime if no explicit `date` variable is provided!"
        date = "date"
        dt = dataset.loc[:, [price]]
        dt[date] = dt.index

    cgr = "cqgr" if quarter else "cygr"
    periods = 3 if quarter else 12

    dt["year"] = dt.loc[:, date].apply(lambda x: x.year)
    dt["month"] = dt.loc[:, date].apply(lambda x: x.month)

    dt[cgr] = (dt[price] / dt.loc[:, price].shift(periods)) ** (1 / periods) - 1

    min_year = int(dt.loc[dt.loc[:, cgr].notna(), "year"].min())
    years = list(dt.year.unique())

    dt["tmp_weight"] = dt.year - min_year + 1
    dt["tmp_nweight"] = dt.tmp_weight - 1

    dt[f"wmean({cgr})"] = np.nan
    dt["tmp_wdata"] = dt.tmp_weight * dt.loc[:, cgr].fillna(0)

    for m in range(1, 13):
        mask = dt.month == m
        dt.loc[mask, f"wmean({cgr})"] = (
            dt.loc[mask, "tmp_wdata"].cumsum() - dt.loc[mask, "tmp_wdata"]
        ) / (dt.loc[mask, "tmp_weight"].cumsum() - dt.loc[mask, "tmp_weight"])

    dt.loc[dt.loc[:, cgr].shift(12).isna(), f"wmean({cgr})"] = np.nan

    dt[f"wsd({cgr})"] = np.nan
    for m in range(1, 13):
        for y in years:
            mask = dt.month == m
            ymask = (dt.year == y) & (dt.month == m)

            try:
                mean_val = float(dt.loc[ymask, f"wmean({cgr})"].iloc[0])
            except IndexError:
                continue
            dt.loc[mask, "tmp_wdata"] = (
                dt.loc[mask, "tmp_weight"]
                * (dt.loc[mask, cgr].fillna(0) - mean_val) ** 2
            )
            dt.loc[mask, "tmp_wdata"] = dt.loc[mask].tmp_wdata.fillna(
                0
            ).cumsum() - dt.loc[mask].tmp_wdata.fillna(0)
            dt.loc[mask, "tmp_cweight"] = (
                dt.loc[mask].tmp_weight.cumsum() - dt.loc[mask].tmp_weight
            )
            dt.loc[ymask, f"wsd({cgr})"] = np.sqrt(
                dt.loc[ymask].tmp_wdata
                / (
                    dt.loc[ymask].tmp_cweight
                    * (dt.loc[ymask].tmp_nweight - 1)
                    / dt.loc[ymask].tmp_nweight
                )
            )

    dt[f"ipa({cgr})"] = (dt[cgr] - dt[f"wmean({cgr})"]) / dt[f"wsd({cgr})"]

    return dt.loc[:, [cgr, f"ipa({cgr})"]]


def fao_procedure(
    dataset: pd.DataFrame,
    price: str | None = None,
    date: str | None = None,
) -> dict[str, str | pd.DataFrame | None]:
    """
    Данная функция рассчитывает сводный Индикатор Ценовых Аномалий (IPA).
    Сводный индикатор является взвешенным средним квартального и годового индикаторов.

    Parameters
    ----------
    dataset : pd.DataFrame
        Источник данных для расчета IPA.

    price : str
        Имя столбца в данных, содержащего данные по ценам.

    date : str | None
        Имя столбца, содержащего дату каждого наблюдения.
        Дата должна быть в формате, содержащем атрибуты `year` и `month`.

        Если имя переменной не задано, то функция пытается использовать индекс датасета.
        В этом случае индекс должен быть типа `DatetimeIndex`.

    Returns
    _______
    dict
        Словарь, содержащий имена переменных с ценами и датами, а также датафрейм с результатами расчетов.

        В датафрейм, помимо исходных переменных, входят столбцы:
        - `UN_ipa`: сводный IPA, рассчитанный по весам, предложенным в официальной документации ООН.
        - `UN_result`: представление `UN_ipa` в тестовом виде. Принимает значения `Alert low` (опасное снижение),
          `Watch low` (беспокоящее снижение), `Normal` (в пределах нормы), `Watch high` (беспокоящий рост),
          `Alert high` (опасный рост).
        - `PC_ipa`, `PC_result`: аналогичные показатели, но использующие веса, рассчитанные при помощи собственных значений
          ковариационной матрицы т.н. композитных квартальных и годовых доходностей.

    Raises
    ------
    AssertionError
        Что-то не то со входными данными.
    """
    data = dataset.copy()

    res_q = get_ipa(data, price, date, quarter=True)
    data = data.merge(res_q, left_index=True, right_index=True)

    res_y = get_ipa(data, price, date, quarter=False)
    data = data.merge(res_y, left_index=True, right_index=True)

    eigen = np.linalg.eig(data.loc[:, ["cqgr", "cygr"]].dropna().cov())[0]
    weights = eigen / sum(eigen)

    data["UN_ipa"] = 0.4 * data["ipa(cqgr)"] + 0.6 * data["ipa(cygr)"].fillna(0)
    data["UN_result"] = data["UN_ipa"].apply(lambda x: ipa_ranges(x))

    data["PC_ipa"] = weights[0] * data[f"ipa(cqgr)"] + weights[1] * data[
        f"ipa(cygr)"
    ].fillna(0)
    data["PC_result"] = data["PC_ipa"].apply(lambda x: ipa_ranges(x))

    return {"price": price, "date": date, "data": data}



class FaoAdapter:
    @staticmethod
    def validate_data(result) -> list[Optional[FaoEnum]]:
        validated = []
        for v in result:
            match v:
                case "Alert low" | "Watch low" | "Normal" | "Watch high" | "Alert high":
                    validated.append(FaoEnum(v))
                case _:
                    validated.append(None)
        return validated

    def run(self, ts: pd.DataFrame) -> FaoResult:
        try:
            fao_res = fao_procedure(ts, price=ts.columns[0], date=None)
            fao_data = fao_res["data"]
        except Exception as e:
            raise HTTPException(status_code=400, detail=SmallSizeError().detail)

        return FaoResult(
            UN_result=self.validate_data(fao_data["UN_result"].values),
            PC_result=self.validate_data(fao_data["PC_result"].values),
        )
