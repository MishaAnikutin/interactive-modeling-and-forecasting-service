import time
from datetime import datetime
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
    def split_ts(
        ts: pd.Series | pd.DataFrame, train_boundary: datetime, val_boundary: datetime
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Разбивает Series без перекрытия границ."""
        # преобразуем в datetime
        idx = pd.to_datetime(ts.index)

        # Удалим временную зону
        train_boundary = train_boundary.replace(tzinfo=None)
        val_boundary = val_boundary.replace(tzinfo=None)

        train_mask = idx <= train_boundary
        val_mask = (idx > train_boundary) & (idx <= val_boundary)
        test_mask = idx > val_boundary

        train_target = ts.loc[train_mask]
        val_target = ts.loc[val_mask]
        test_target = ts.loc[test_mask]
        return train_target, val_target, test_target

    def split(
        self,
        train_boundary: datetime,
        val_boundary: datetime,
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

        # --- целевая переменная ---
        train_target, val_target, test_target = self.split_ts(
            target, train_boundary, val_boundary
        )

        # --- экзогенные признаки (если есть) ---
        if exog is not None:
            exog_train, exog_val, exog_test = self.split_ts(
                exog, train_boundary, val_boundary
            )
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
