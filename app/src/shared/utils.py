import math
from typing import Optional


def validate_float_param(param: Optional[float]) -> Optional[float]:
    if param is not None and math.isfinite(param):
        return round(param, 4)
    return None
