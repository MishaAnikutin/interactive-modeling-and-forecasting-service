from enum import Enum


class GrowthConclusion(str, Enum):
    """Заключение о характере роста"""
    normal = "Рост в норме"
    anomalous = "Рост аномален"
