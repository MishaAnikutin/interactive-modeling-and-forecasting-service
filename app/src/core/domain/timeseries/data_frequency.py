from enum import Enum


class DataFrequency(str, Enum):
    year = "Y"
    month = "M"
    quart = "Q"
    day = "D"
    hour = "H"
    minute = "M"
