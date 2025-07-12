import numpy as np
import pandas as pd
from fastapi import HTTPException
from scipy.stats import boxcox

from src.core.application.preprocessing.preprocess_scheme import (
    DiffTransformation,
    LagTransformation,
    LogTransformation,
    PowTransformation,
    NormalizeTransformation,
    ExpSmoothTransformation,
    BoxCoxTransformation,
    FillMissingTransformation,
    MovingAverageTransformation,
)
from src.core.domain.preprocessing.service import PreprocessingServiceI
from src.infrastructure.adapters.preprocessing.preprocess_factory import PreprocessFactory


@PreprocessFactory.register(transform_type="diff")
class Diff(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: DiffTransformation) -> pd.Series:
        return ts.diff(periods=transformation.diff_order)


@PreprocessFactory.register(transform_type="lag")
class Lag(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: LagTransformation) -> pd.Series:
        return ts.shift(periods=transformation.lag_order)


@PreprocessFactory.register(transform_type="log")
class Log(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: LogTransformation) -> pd.Series:
        safe_ts = ts.copy()
        return np.log(safe_ts)


@PreprocessFactory.register(transform_type="pow")
class Pow(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: PowTransformation) -> pd.Series:
        exp = transformation.pow_order
        if float(exp).is_integer():
            exp = int(exp)
        return ts ** exp


@PreprocessFactory.register(transform_type="normalize")
class Normalize(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: NormalizeTransformation) -> pd.Series:
        if transformation.method == "minmax":
            return (ts - ts.min()) / (ts.max() - ts.min())
        return (ts - ts.mean()) / ts.std(ddof=0)


@PreprocessFactory.register(transform_type="exp_smooth")
class ExpSmooth(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: ExpSmoothTransformation) -> pd.Series:
        return ts.ewm(span=transformation.span).mean()


@PreprocessFactory.register(transform_type="boxcox")
class BoxCox(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: BoxCoxTransformation) -> pd.Series:
        safe_ts = ts.copy()
        if (safe_ts <= 0).any():
            raise HTTPException(
                status_code=400,
                detail="Все элементы ряда должны быть положительные для выполнения преобразования Бокса-Кокса"
            )
        transformed = boxcox(safe_ts, lmbda=transformation.param)
        return pd.Series(transformed, index=ts.index, name=ts.name)


@PreprocessFactory.register(transform_type="fillna")
class FillNa(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: FillMissingTransformation) -> pd.Series:
        if transformation.method == "last":
            return ts.ffill()
        if transformation.method == "backward":
            return ts.bfill()
        if transformation.method == "mean":
            return ts.fillna(ts.mean())
        if transformation.method == "median":
            return ts.fillna(ts.median())
        return ts.fillna(ts.mode().iloc[0] if not ts.mode().empty else ts)


@PreprocessFactory.register(transform_type="moving_avg")
class MovingAverage(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: MovingAverageTransformation) -> pd.Series:
        return ts.rolling(window=transformation.window, min_periods=1).mean()