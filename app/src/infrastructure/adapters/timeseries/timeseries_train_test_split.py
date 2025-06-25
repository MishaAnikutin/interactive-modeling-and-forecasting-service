from datetime import datetime
from typing import Tuple, TypeVar, Optional

Xtype = TypeVar("Xtype")
Ytype = TypeVar("Ytype")


class TimeseriesTrainTestSplit:
    """
    Делит временной ряд на обучающую, валидационную и тестовую части.

    train_boundary – последний момент, попадающий в train.
    val_boundary   – последний момент, попадающий в validation.
                     Всё, что идёт после val_boundary, считается test-частью.
    """
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
        train_target = target.loc[:train_boundary]
        val_target = target.loc[train_boundary:val_boundary]
        test_target = target.loc[val_boundary:]

        # --- экзогенные признаки (если есть) ---
        if exog is not None:
            exog_train = exog.loc[:train_boundary]
            exog_val = exog.loc[train_boundary:val_boundary]
            exog_test = exog.loc[val_boundary:]
        else:
            exog_train = exog_val = exog_test = None

        return (
            exog_train,
            train_target,
            exog_val,
            val_target,
            exog_test,
            test_target,
        )