from math import ceil, pi, sqrt
from typing import Literal
from warnings import simplefilter

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
from statsmodels.api import OLS

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Таблицы для расчета критических и p-значений взяты из пакета `uroot` для языка R.
# López-de-Lacalle J (2023).
# uroot: Unit Root Tests for Seasonal Time Series.
# R package version 2.1-3, https://geobosh.github.io/uroot/.
#
# Так как в пакете не представлены таблицы для случая отсутствия детерминированных
# компонент, они были рассчитаны авторами на основе исходной методики
# Ignacio Diaz-Emparanza,
# Numerical distribution functions for seasonal unit root tests,
# Computational Statistics & Data Analysis, Volume 76, 2014, Pages 237-247.
# ISSN 0167-9473, https://doi.org/10.1016/j.csda.2013.03.006.
#
# Использовался сокращенный FFD подход на основе использования латинского квадрата.
TYPES = {
    "zero": "Ct1",
    "pi": "Ct2",
    "pair": "CF",
    "seas": "CFs",
    "all": "CFt",
}


def _aux_regressors(y: np.ndarray, S: int) -> np.ndarray:
    evenS = 1 - (S % 2)
    Star = S // 2

    rows = len(y)
    result = np.full((rows, S), np.nan)

    mlags = np.full((rows, S), np.nan)
    mlags[:, 0] = y[:]
    for lag in range(1, S):
        mlags[lag:, lag] = y[:-lag]

    mcos = np.cos(np.arange(0, S) * 2 * pi / S)
    msin = -np.sin(np.arange(0, S) * 2 * pi / S)

    result[1:, 0] = (mlags @ np.ones(S))[:-1]
    if evenS:
        result[1:, S - 1] = (
            mlags @ np.take(mcos, Star * np.arange(1, S + 1), mode="wrap")
        )[:-1]

    for k in range(1, Star - evenS + 1):
        result[1:, 2 * k - 1] = (
            mlags @ np.take(mcos, k * np.arange(1, S + 1), mode="wrap")
        )[:-1]
        result[1:, 2 * k] = (
            mlags @ np.take(msin, k * np.arange(1, S + 1), mode="wrap")
        )[:-1]

    return result


def _seasonal_dummies(rows: int, S: int) -> np.ndarray:
    return np.tile(np.eye(S), ceil(rows / S))[1:, :rows].T


def seasonalURoot(
    y: np.ndarray | pd.Series,
    max_lag: int | None = None,
    trend: str = "n",
    criteria: str = "aic",
    S: int = 4,
    stats_only: bool = False,
):
    if max_lag is None:
        max_lag = int(12 * ((len(y) / 100) ** (1 / 4)))

    assert trend in (
        "n",
        "c",
        "cd",
        "ct",
        "cdt",
    ), "Неправильная спецификация детерминированной компоненты!"
    assert criteria in ("aic", "bic", "hqic", "fixed"), "Неправильный IC!"

    evenS = 1 - (S % 2)
    Star = S // 2

    _offset = (
        S
        + (1 if trend != "n" else 0)
        + (S - 1 if "d" in trend else 0)
        + (1 if "t" in trend else 0)
    )

    rows = len(y)
    cols = _offset + max_lag + 1
    datamat = np.full((rows, cols), np.nan)

    y = np.asarray(y)

    datamat[S:, 0] = y[S:] - y[:-S]
    datamat[:, 1 : (S + 1)] = _aux_regressors(y, S)

    if trend != "n":
        datamat[:, S + 1] = 1
    if "d" in trend:
        datamat[:, (S + 2) : (2 * S + 1)] = _seasonal_dummies(rows, S)
    if "t" in trend:
        datamat[:, _offset] = np.arange(0, rows) - S

    for lag in range(1, max_lag + 1):
        datamat[lag:, _offset + lag] = datamat[:-lag, 0]

    datamat = datamat[~np.isnan(datamat).any(axis=1)]
    _Y, _X = np.hsplit(datamat, [1])
    del datamat

    ic_model = None
    ic_value = float("inf")

    if criteria != "fixed" and max_lag > 0:
        for lag in range(max_lag + 1):
            loop_model = OLS(_Y, _X[:, : (_offset + lag)]).fit()
            loop_value = loop_model.info_criteria(criteria)

            if loop_value < ic_value:
                ic_value = loop_value
                ic_model = loop_model
            else:
                break
    else:
        ic_model = OLS(_Y, _X).fit()

    assert ic_model is not None, "Что-то пошло не так, модель не подобрана!"

    result = pd.DataFrame({"stat": np.empty(Star + 3), "pval": np.empty(Star + 3)})
    result.index = (
        ["Zero freq"]
        + (["Nyquist"] if evenS else [])
        + [f"{i}w" for i in range(1, Star - evenS + 1)]
        + ["Seas.", "ALL"]
    )

    _common = {
        "T": rows,
        "S": S,
        "lags": max_lag,
        "trend": trend,
        "criteria": criteria,
    }

    cols = ic_model.params.size

    rmatrix = np.zeros(cols)
    rmatrix[0] = 1
    result.loc["Zero freq", "stat"] = ic_model.t_test(rmatrix).tvalue
    if evenS:
        rmatrix = np.zeros(cols)
        rmatrix[S - 1] = 1
        result.loc["Nyquist", "stat"] = ic_model.t_test(rmatrix).tvalue

    for k in range(1, Star - evenS + 1):
        rmatrix = np.zeros((2, cols))
        rmatrix[:, (2 * k - 1) : (2 * k + 1)] = np.eye(2)
        result.loc[f"{k}w", "stat"] = ic_model.f_test(rmatrix).fvalue

    rmatrix = np.zeros((S - 1, cols))
    rmatrix[:, 1:S] = np.eye(S - 1)
    result.loc["Seas.", "stat"] = ic_model.f_test(rmatrix).fvalue

    rmatrix = np.zeros((S, cols))
    rmatrix[:, :S] = np.eye(S)
    result.loc["ALL", "stat"] = ic_model.f_test(rmatrix).fvalue

    if not stats_only:
        result.loc["Zero freq", "pval"] = seasonalURootCV(
            x=result.loc["Zero freq", "stat"], test="zero", **_common
        )
        if evenS:
            result.loc["Nyquist", "pval"] = seasonalURootCV(
                x=result.loc["Nyquist", "stat"], test="pi", **_common
            )
        for k in range(1, Star - evenS + 1):
            result.loc[f"{k}w", "pval"] = seasonalURootCV(
                x=result.loc[f"{k}w", "stat"], test="pair", **_common
            )
        result.loc["Seas.", "pval"] = seasonalURootCV(
            x=result.loc["Seas.", "stat"], test="seas", **_common
        )
        result.loc["ALL", "pval"] = seasonalURootCV(
            x=result.loc["ALL", "stat"], test="all", **_common
        )

    return result


def seasonalURootCV(
    x: float,
    T: int,
    S: int,
    lags: int,
    test: Literal["zero", "pi", "pair", "seas", "all"] = "zero",
    trend: Literal["n", "c", "cd", "ct", "cdt"] = "c",
    criteria: Literal["aic", "bic", "fixed"] = "fixed",
    regT: int = 15,
):
    # data = rr.read_r(f"hegy_cv/{TYPES[test]}_{trend}_{criteria}.Rds")[None].values
    data = np.load(f"src/shared/hegy/hegy_coefs/{TYPES[test]}_{trend}_{criteria}.npy")

    isFtest = test not in ("zero", "pi")
    match test:
        case "pair":
            df = 2
        case "seas":
            df = S - 1
        case "all":
            df = S
        case _:
            df = 0

    nd = int("c" in trend) + (S - 1) * int("d" in trend) + int("t" in trend)

    coefs = data[:, :-1]
    sd = data[:, -1]
    T = T - S - lags - nd

    Ps = np.hstack(
        [
            (0.0001, 0.0002, 0.0005),
            np.linspace(0.001, 0.010, 10),
            np.linspace(0.015, 0.985, 195),
            np.linspace(0.990, 0.999, 10),
            (0.9995, 0.9998, 0.9999),
        ]
    )
    nPs = len(Ps)

    vars = np.array(
        [
            1,
            1 / T,
            1 / T**2,
            1 / T**3,
            lags / T,
            lags / T**2,
            lags / T**3,
            lags**2 / T,
            lags**2 / T**2,
            lags**2 / T**3,
            lags**3 / T,
            lags**3 / T**2,
            lags**3 / T**3,
            S / T,
            S / T**2,
            S / T**3,
        ]
    )

    Qs = coefs @ vars
    idx = np.argsort(Qs)
    Qs = Qs[idx]
    sd = sd[idx]

    if x < Qs.min():
        return 1.0 if isFtest else 0.0
    elif x > Qs.max():
        return 0.0 if isFtest else 1.0

    try:
        mask = np.where(x > Qs)[0].max()
    except ValueError:
        mask = 0

    if mask < nPs - 1:
        mask += 0 if (x - Qs[mask]) < (Qs[mask + 1] - x) else 1
    else:
        mask = nPs - 1

    mask = max(mask, regT // 2)
    mask = min(mask, nPs - regT // 2 - 1)

    regQ = Qs[(mask - regT // 2) : (mask - regT // 2 + regT)]
    regP = Ps[(mask - regT // 2) : (mask - regT // 2 + regT)]
    regSD = sd[(mask - regT // 2) : (mask - regT // 2 + regT)]

    Y = chi2.ppf(regP, df=df) if isFtest else norm.ppf(regP)
    X = np.vstack([np.ones(regT), regQ, regQ**2, regQ**3]).T

    Sigma = np.empty((regT, regT))
    for i in range(regT):
        for j in range(regT):
            if i <= j:
                Sigma[i, j] = (regSD[i] * regSD[j]) * sqrt(
                    (regP[i] * (1 - regP[j])) / (regP[j] * (1 - regP[i]))
                )
            else:
                Sigma[i, j] = (regSD[i] * regSD[j]) * sqrt(
                    (regP[j] * (1 - regP[i])) / (regP[i] * (1 - regP[j]))
                )

    Chol = np.linalg.inv(np.linalg.cholesky(Sigma, upper=True)).T

    X = Chol @ X
    Y = Chol @ Y

    coefP = np.linalg.solve(X.T @ X, X.T @ Y)
    value = np.dot(coefP, (1, x, x**2, x**3))

    return chi2.cdf(abs(value), df=df) if isFtest else norm.cdf(value)
