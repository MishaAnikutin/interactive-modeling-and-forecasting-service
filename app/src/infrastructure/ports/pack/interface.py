import abc


class ModelPacker(abc.ABC):
    """Упаковывает результаты обучения модели по инфраструктурным правилам"""
    @abc.abstractmethod
    def pack(self):
        ...
