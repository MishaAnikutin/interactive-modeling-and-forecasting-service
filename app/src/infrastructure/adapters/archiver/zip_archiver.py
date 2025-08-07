import json
import zipfile
from io import BytesIO

from .interface import ModelArchiver


class ZipArchiver(ModelArchiver):
    def execute(self, data_dict: dict, model_bytes: bytes) -> bytes:
        """Создает ZIP-архив, содержащий JSON-файл с данными и .pickle файл с моделью."""

        archive_buffer = BytesIO()

        with zipfile.ZipFile(archive_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            json_data = json.dumps(data_dict, default=str, ensure_ascii=False, indent=4).encode('utf-8')
            zipf.writestr('fit_results.json', json_data)
            zipf.writestr('model.pickle', model_bytes)

        archive_buffer.seek(0)

        return archive_buffer.getvalue()
