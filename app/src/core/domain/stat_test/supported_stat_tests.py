"""
Какие стат тесты реализованы в фабрике
"""
from enum import Enum


class SupportedStationaryTests(str, Enum):
    DickeyFuller: str = 'DickeyFuller'
    KPSS: str = 'KPSS'
