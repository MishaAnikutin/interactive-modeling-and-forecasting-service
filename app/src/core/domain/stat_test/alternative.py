from enum import Enum


class Alternative(str, Enum):
    TWO_SIDED = "two-sided"
    GREATER = "greater"
    LESS = "less"
