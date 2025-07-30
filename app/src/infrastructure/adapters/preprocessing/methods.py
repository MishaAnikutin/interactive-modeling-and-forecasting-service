from typing import Tuple, Optional

import numpy as np
import pandas as pd
from scipy.stats import boxcox

from src.core.application.preprocessing.preprocess_scheme import (
    DiffTransformation,
    LagTransformation,
    LogTransformation,
    PowTransformation,
    MinMaxTransformation,
    StandardTransformation,
    ExpSmoothTransformation,
    BoxCoxTransformation,
    FillMissingTransformation,
    MovingAverageTransformation,
    InverseMinMaxTransformation,
    InverseStandardTransformation,
    InverseDiffTransformation,
    InverseLagTransformation,
    InverseLogTransformation,
    InversePowTransformation,
    InverseExpSmoothTransformation,
    InverseBoxCoxTransformation,
    InverseFillMissingTransformation,
    InverseMovingAverageTransformation,
    DiffContext,
    MinMaxContext,
    StandardContext,
)
from src.core.domain.preprocessing.service import PreprocessingServiceI
from src.infrastructure.adapters.preprocessing.preprocess_factory import (
    PreprocessFactory,
)


@PreprocessFactory.register(transform_type="diff")
class Diff(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: DiffTransformation) -> Tuple[pd.Series, DiffContext]:
        return ts.diff(periods=transformation.diff_order), DiffContext(first_values=ts[:transformation.diff_order])

    def inverse(
        self, ts: pd.Series, transformation: InverseDiffTransformation
    ) -> pd.Series:
        periods = transformation.diff_order
        restored = np.full(len(ts), np.nan)
        restored[:periods] = transformation.first_values

        for i in range(periods, len(ts)):
            restored[i] = restored[i - periods] + ts.iloc[i]

        return pd.Series(restored, index=ts.index, name=ts.name)


@PreprocessFactory.register(transform_type="lag")
class Lag(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: LagTransformation) -> Tuple[pd.Series, None]:
        return ts.shift(periods=transformation.lag_order), None

    def inverse(
        self, ts: pd.Series, transformation: InverseLagTransformation
    ) -> pd.Series:
        return ts.shift(periods=-transformation.lag_order)


@PreprocessFactory.register(transform_type="log")
class Log(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: LogTransformation) -> Tuple[pd.Series, None]:
        safe_ts = ts.copy()  # FIXME: зачем копию делать ... ?
        return np.log(safe_ts), None

    def inverse(
        self, ts: pd.Series, transformation: InverseLogTransformation
    ) -> pd.Series:
        return np.exp(ts)


@PreprocessFactory.register(transform_type="pow")
class Pow(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: PowTransformation) -> Tuple[pd.Series, None]:
        exp = transformation.pow_order
        if float(exp).is_integer():  # FIXME: ...?
            exp = int(exp)
        return ts ** exp, None

    def inverse(
        self, ts: pd.Series, transformation: InversePowTransformation
    ) -> pd.Series:
        if transformation.pow_order == 0:
            raise TypeError("Невозможно восстановить ряд")

        return ts ** (1 / transformation.pow_order)


@PreprocessFactory.register(transform_type="minmax")
class MinMax(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: MinMaxTransformation) -> Tuple[pd.Series, MinMaxContext]:
        return (ts - ts.min()) / (ts.max() - ts.min()), MinMaxContext(init_min=ts.min(), init_max=ts.max())

    def inverse(
        self, ts: pd.Series, transformation: InverseMinMaxTransformation
    ) -> pd.Series:
        return (
            ts * (transformation.init_max - transformation.init_min)
            + transformation.init_min
        )


@PreprocessFactory.register(transform_type="standard")
class Standard(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: StandardTransformation) -> Tuple[pd.Series, StandardContext]:
        return (ts - ts.mean()) / ts.std(ddof=0), StandardContext(init_mean=ts.mean(), init_std=ts.std(ddof=0))

    def inverse(
        self, ts: pd.Series, transformation: InverseStandardTransformation
    ) -> pd.Series:
        return ts * transformation.init_std + transformation.init_mean


@PreprocessFactory.register(transform_type="exp_smooth")
class ExpSmooth(PreprocessingServiceI):
    def apply(
        self, ts: pd.Series, transformation: ExpSmoothTransformation
    ) -> Tuple[pd.Series, None]:
        return ts.ewm(span=transformation.span).mean(), None

    def inverse(
        self, ts: pd.Series, transformation: InverseExpSmoothTransformation
    ) -> pd.Series:
        # Для алгоритмов н аскользящем окне возвращаем ряд без изменений
        return ts


@PreprocessFactory.register(transform_type="boxcox")
class BoxCox(PreprocessingServiceI):
    def apply(self, ts: pd.Series, transformation: BoxCoxTransformation) -> Tuple[pd.Series, None]:
        if not all(ts >= 0):
            raise ValueError('Для преобразования Бокса-Кокса все значения ряда должны быть больше нуля')

        safe_ts = ts.copy()  # FIXME: зачем копию делать ... ?
        transformed = boxcox(safe_ts, lmbda=transformation.param)
        return pd.Series(transformed, index=ts.index, name=ts.name), None

    def inverse(
        self, ts: pd.Series, transformation: InverseBoxCoxTransformation
    ) -> pd.Series:
        if transformation.param == 0:
            return np.exp(ts)

        return (transformation.param * ts + 1) ** (1 / transformation.param)


@PreprocessFactory.register(transform_type="fillna")
class FillNa(PreprocessingServiceI):
    def apply(
        self, ts: pd.Series, transformation: FillMissingTransformation
    ) -> Tuple[pd.Series, None]:
        if transformation.method == "last":
            return ts.ffill(), None
        if transformation.method == "backward":
            return ts.bfill(), None
        if transformation.method == "mean":
            return ts.fillna(ts.mean()), None
        if transformation.method == "median":
            return ts.fillna(ts.median()), None
        return ts.fillna(ts.mode().iloc[0] if not ts.mode().empty else ts), None

    def inverse(
        self, ts: pd.Series, transformation: InverseFillMissingTransformation
    ) -> pd.Series:
        # Очевидно, обратно предобрабатывать нет смысла
        return ts


@PreprocessFactory.register(transform_type="moving_avg")
class MovingAverage(PreprocessingServiceI):
    def apply(
        self, ts: pd.Series, transformation: MovingAverageTransformation
    ) -> Tuple[pd.Series, None]:
        return ts.rolling(window=transformation.window, min_periods=1).mean(), None

    def inverse(
        self, ts: pd.Series, transformation: InverseMovingAverageTransformation
    ) -> pd.Series:
        # Для алгоритмов н аскользящем окне возвращаем ряд без изменений
        return ts
