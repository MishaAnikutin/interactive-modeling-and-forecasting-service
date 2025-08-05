import time
from abc import abstractmethod, ABC


class IModelStorage(ABC):
    @abstractmethod
    def save(self, model_result) -> tuple[str, str]:
        return "заглушка", "заглушка"


class MockModelStorage(IModelStorage):
    def save(self, arimax_fit_result) -> tuple[str, str]:
        print("Сохраняем модель!")
        time.sleep(1)
        return "заглушка", "заглушка"