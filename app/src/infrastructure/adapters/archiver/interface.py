import abc


class ModelArchiver(abc.ABC):
    @abc.abstractmethod
    def execute(self, data_dict: dict, model_bytes: bytes) -> bytes:
        ...