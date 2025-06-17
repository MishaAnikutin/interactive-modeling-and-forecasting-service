from datetime import datetime
from typing import Tuple, TypeVar

Xtype = TypeVar("Xtype")
Ytype = TypeVar("Ytype")


class TimeseriesTrainTestSplit:
    def split(
        self,
        train_boundary: datetime,
        target: Ytype,
        exog: Xtype,
    ) -> Tuple[Xtype, Ytype, Xtype, Ytype]:

        train_target = target.iloc[:train_boundary]
        test_target = target.iloc[train_boundary:]

        if exog is not None:
            exog_train = exog.iloc[:train_boundary]
            exog_test = exog.iloc[train_boundary:]
        else:
            exog_train = None
            exog_test = None

        return exog_train, train_target, exog_test, test_target
