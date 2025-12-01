class TwoSigmaTestError(ValueError):
    ...


class InvalidDateError(TwoSigmaTestError):
    ...


class InsufficientDataError(TwoSigmaTestError):
    ...
