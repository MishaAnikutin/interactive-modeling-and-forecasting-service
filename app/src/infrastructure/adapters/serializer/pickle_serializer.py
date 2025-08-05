import base64
import pickle
from .interface import ModelSerializer, T


class PickleSerializer(ModelSerializer):
    def serialize(self, model: T) -> bytes:
        return pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)

    def undo_serialize(self, serialized_model: bytes) -> T:
        return pickle.loads(serialized_model)


# FIXME: это костыль пока не придумали с бэкендерами как
#  возвращать .pickle файлы в S3 вместе с метриками и прогнозами.
#  Нужна строка чтобы в JSON можно помещать
class Base64PickleSerializer(ModelSerializer):
    def serialize(self, model: T) -> str:
        return base64.b64encode(
            pickle.dumps(model, protocol=pickle.HIGHEST_PROTOCOL)
        ).decode("utf-8")

    def undo_serialize(self, serialized_model: str) -> T:
        return pickle.loads(
            base64.b64decode(serialized_model)
        )
