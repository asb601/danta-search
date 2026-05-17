from datetime import datetime
from pydantic import BaseModel, field_validator


class ContainerCreate(BaseModel):
    name: str
    container_name: str
    connection_string: str
    semantic_config: dict | None = None

    @field_validator("name", "container_name", "connection_string", mode="before")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        return v.strip() if isinstance(v, str) else v


class ContainerOut(BaseModel):
    id: str
    name: str
    container_name: str
    last_synced_at: datetime | None
    file_count: int = 0
    created_at: datetime
    semantic_config: dict | None = None

    model_config = {"from_attributes": True}


class ContainerSemanticConfigUpdate(BaseModel):
    semantic_config: dict | None = None


class ContainerSemanticRebuildRequest(BaseModel):
    re_resolve_roles: bool = True
    batch_size: int = 250


class ContainerSyncResponse(BaseModel):
    message: str
    container_id: str
