from statsmodels.tsa.stattools import range_unit_root_test

from src.core.application.preliminary_diagnosis.schemas.common import CriticalValues
from src.core.application.preliminary_diagnosis.schemas.range_scheme import RangeUnitRootResult, RangeUnitRootParams
from src.infrastructure.adapters.timeseries import PandasTimeseriesAdapter


class RangeUnitRootUC:
    def __init__(
        self,
        ts_adapter: PandasTimeseriesAdapter,
    ):
        self._ts_adapter = ts_adapter

    def execute(self, request: RangeUnitRootParams) -> RangeUnitRootResult:
        ts = self._ts_adapter.to_series(ts_obj=request.ts)
        rur_stat, p_value, crit_dict = range_unit_root_test(x=ts)
        return RangeUnitRootResult(
            p_value=p_value,
            stat_value=rur_stat,
            critical_values=CriticalValues(
                percent_1=crit_dict["1%"],
                percent_5=crit_dict["5%"],
                percent_10=crit_dict["10%"],
            )
        )