import pickle
from .interface import ModelSerializer, T


class PickleSerializer(ModelSerializer):
    def serialize(self, model: T) -> bytes:
        return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

    def deserialize(self, serialized_model: bytes) -> T:
        return pickle.loads(serialized_model)
