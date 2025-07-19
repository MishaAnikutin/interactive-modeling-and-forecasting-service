from typing import Optional


def validate_float_param(param) -> Optional[float]:
    if param is not None:
        if (
                param == float("inf") or
                param == float("-inf") or
                param == float("nan")
        ):
            return None
        else:
            return round(param, 4)
    return None
