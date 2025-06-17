import time
from abc import abstractmethod, ABC


class IModelStorage(ABC):
    @abstractmethod
    def save(self, model_result): ...


class MockModelStorage(IModelStorage):
    def save(self, arimax_fit_result):
        print("Сохраняем модель!")
        time.sleep(1)
