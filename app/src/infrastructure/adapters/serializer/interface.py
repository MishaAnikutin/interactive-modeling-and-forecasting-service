import abc
from typing import Generic, TypeVar


T = TypeVar('T')


class ModelSerializer(abc.ABC, Generic[T]):
    """Сериализует веса моделей в байты"""

    @abc.abstractmethod
    def serialize(self, model: T) -> bytes:
        ...

    @abc.abstractmethod
    def undo_serialize(self, serialized_model: bytes) -> T:
        ...
