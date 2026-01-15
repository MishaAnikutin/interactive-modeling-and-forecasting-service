from enum import Enum


class NanPolicy(str, Enum):
    PROPAGATE = "propagate"
    OMIT = "omit"
    RAISE = "raise"
