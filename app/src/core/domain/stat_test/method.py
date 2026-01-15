from enum import Enum


class Method(str, Enum):
    AUTO = "auto"
    EXACT = "exact"
    ASYMPTOTIC = "asymptotic"
