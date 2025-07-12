import json
import logging
import os
import sys
from logging.config import dictConfig

LOG_LEVEL = os.getenv("LOG_LEVEL", "ERROR").upper()
LOG_FILE = os.getenv("LOG_FILE", "app.log")
SERVICE_NAME = os.getenv("SERVICE_NAME", "ml-service")

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        log_record = {
            "ts": self.formatTime(record, datefmt="%Y-%m-%dT%H:%M:%S.%fZ"),
            "lvl": record.levelname,
            "svc": SERVICE_NAME,
            "msg": record.getMessage(),
            "mod": record.module,
            "fn": record.funcName,
            "ln": record.lineno,
            "request_id": getattr(record, "request_id", None),
            "corr_id": getattr(record, "correlation_id", None),
        }
        if record.exc_info:
            log_record["exc"] = self.formatException(record.exc_info)
        return json.dumps(log_record, ensure_ascii=False)

# dictConfig(
#     {
#         "version": 1,
#         "disable_existing_loggers": False,
#         "formatters": {
#             "json": {
#                 "()": JsonFormatter,
#             },
#             "plain": {
#                 "format": "[%(asctime)s] %(levelname)s: %(message)s",
#             },
#         },
#         "handlers": {
#             "console": {
#                 "class": "logging.StreamHandler",
#                 "formatter": "json",
#                 "stream": sys.stdout,
#             },
#             "file": {
#                 "class": "logging.handlers.TimedRotatingFileHandler",
#                 "formatter": "json",
#                 "filename": LOG_FILE,
#                 "when": "midnight",
#                 "backupCount": 14,
#                 "encoding": "utf-8",
#             },
#         },
#         "root": {
#             "level": LOG_LEVEL,
#             "handlers": ["console", "file"],
#         },
#     }
# )

logger = logging.getLogger(SERVICE_NAME)
logger.info("Логирование инициализировано", extra={"correlation_id": "startup"})