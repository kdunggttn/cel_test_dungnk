from pydantic import BaseModel, ConfigDict
from typing import Coroutine, List, Any
from http import HTTPStatus


class IBaseResponse(BaseModel):
    """Base response interceptor. Formats responses to be sent to the client."""

    statusCode: HTTPStatus
    message: str | None = None
    data: Any | None = None
    error: str | None = None
