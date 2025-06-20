from uuid import uuid4
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from logs import logger

REQUEST_ID_HEADER = "X-Request-ID"
CORRELATION_ID_HEADER = "X-Correlation-ID"


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid4()))
        corr_id = request.headers.get(CORRELATION_ID_HEADER, request_id)

        request.state.request_id = request_id
        request.state.correlation_id = corr_id

        logger.info(
            "HTTP %s %s", request.method, request.url.path,
            extra={"request_id": request_id, "correlation_id": corr_id},
        )

        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[CORRELATION_ID_HEADER] = corr_id
        return response