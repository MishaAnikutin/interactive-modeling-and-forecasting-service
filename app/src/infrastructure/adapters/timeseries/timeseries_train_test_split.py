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
            ts: pd.Series,
            train_boundary: datetime,
            val_boundary: datetime
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Разбивает Series без перекрытия границ."""
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
        train_boundary: datetime,
        val_boundary: datetime,
        target: Ytype,
        exog: Optional[Xtype] = None,
    ) -> Tuple[
        Optional[Xtype], Ytype,           # train exog / target
        Optional[Xtype], Ytype,           # val   exog / target
        Optional[Xtype], Ytype            # test  exog / target
    ]:

        # --- целевая переменная ---
        train_target, val_target, test_target = self.split_ts(target, train_boundary, val_boundary)

        # --- экзогенные признаки (если есть) ---
        exog_train = exog_val = exog_test = None
        if exog is not None:
            exog_train, exog_val, exog_test = self.split_ts(exog, train_boundary, val_boundary)

        return (
            exog_train,
            train_target,
            exog_val,
            val_target,
            exog_test,
            test_target,
        )