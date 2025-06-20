from enum import Enum


class DataFrequency(str, Enum):
    year: str = "Y"
    month: str = "M"
    quart: str = "Q"
    day: str = "D"
    hour: str = "H"
    minute: str = "M"
