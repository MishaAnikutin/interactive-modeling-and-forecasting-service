from datetime import date
from typing import Tuple, TypeVar, Optional

import pandas as pd

Xtype = TypeVar("Xtype")
Ytype = TypeVar("Ytype")


class TimeseriesTrainTestSplit:
    """
    Делит временной ряд на обучающую, валидационную и тестовую части.

    Правила разбиения
    -----------------
    train_boundary – последний момент, попадающий в train  (<=).
    val_boundary   – последний момент, попадающий в val   (<=).
                     Всё, что строго позже val_boundary, относится к test.
    ┌────────────┬──────────────────────────────┐
    │   train    │ ts.index <= train_boundary   │
    ├────────────┼──────────────────────────────┤
    │    val     │ train_boundary < ts.index <= val_boundary │
    ├────────────┼──────────────────────────────┤
    │    test    │ ts.index  >  val_boundary    │
    └────────────┴──────────────────────────────┘
    """

    @staticmethod
    def _ensure_datetime_index(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """Преобразует индекс в datetime64[ns] при необходимости."""
        if not isinstance(data.index, pd.DatetimeIndex):
            data = data.copy()
            data.index = pd.to_datetime(data.index)
        return data

    def split_ts(
            self,
            ts: pd.Series | pd.DataFrame,
            train_boundary: date,
            val_boundary: date
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Разбивает Series/DataFrame без перекрытия границ."""
        # Преобразуем границы в pd.Timestamp
        train_boundary= pd.Timestamp(train_boundary)
        val_boundary = pd.Timestamp(val_boundary)

        ts = self._ensure_datetime_index(ts)
        idx = ts.index

        train_mask = idx <= train_boundary
        val_mask = (idx > train_boundary) & (idx <= val_boundary)
        test_mask = idx > val_boundary

        train_target = ts.loc[train_mask]
        val_target = ts.loc[val_mask]
        test_target = ts.loc[test_mask]
        return train_target, val_target, test_target

    def split(
            self,
            train_boundary: date,
            val_boundary: date,
            target: pd.Series,
            exog: Optional[pd.DataFrame] = None,
    ) -> Tuple[
        Optional[pd.DataFrame],
        pd.Series,  # train exog / target
        Optional[pd.DataFrame],
        pd.Series,  # val   exog / target
        Optional[pd.DataFrame],
        pd.Series,  # test  exog / target
    ]:
        # Гарантируем datetime индекс для target
        target = self._ensure_datetime_index(target)

        # --- целевая переменная ---
        train_target, val_target, test_target = self.split_ts(
            target, train_boundary, val_boundary
        )

        # --- экзогенные признаки (если есть) ---
        if exog is not None:
            exog = self._ensure_datetime_index(exog)
            exog_train, exog_val, exog_test = self.split_ts(
                exog, train_boundary, val_boundary
            )
            assert exog_train.shape[0] == train_target.shape[0]
            assert exog_val.shape[0] == val_target.shape[0]
            assert exog_test.shape[0] == test_target.shape[0]
            return (
                exog_train,
                train_target,
                exog_val,
                val_target,
                exog_test,
                test_target,
            )

        return (
            None,
            train_target,
            None,
            val_target,
            None,
            test_target,
        )
