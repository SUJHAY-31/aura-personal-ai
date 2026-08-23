"""Public health and identity endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..utils.metadata import APP_NAME, APP_VERSION

router = APIRouter(tags=["Health"])


class RootResponse(BaseModel):
    """Payload returned by the service root endpoint."""

    assistant: str = Field(
        default=APP_NAME,
        description="Assistant product identifier.",
        examples=["AURA"],
    )
    status: str = Field(
        default="Running",
        description="Current runtime status of the API process.",
        examples=["Running"],
    )
    version: str = Field(
        default=APP_VERSION,
        description="Semantic version of the running API.",
        examples=["0.1.0"],
    )
    message: str = Field(
        default="Welcome to AURA AI",
        description="Human-readable welcome message.",
        examples=["Welcome to AURA AI"],
    )


@router.get(
    "/",
    response_model=RootResponse,
    summary="Service identity and health",
    response_description="Assistant name, status, version, and welcome message.",
)
def read_root() -> RootResponse:
    """Confirm the API is up and return AURA identity metadata."""
    return RootResponse()
