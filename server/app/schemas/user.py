from datetime import datetime
from pydantic import BaseModel


class UserOut(BaseModel):
    id: str
    email: str
    name: str | None
    picture: str | None
    is_admin: bool
    role: str = "user"
    created_at: datetime
    file_count: int = 0
    allowed_domains: list[str] | None = None
    organization_id: str | None = None

    model_config = {"from_attributes": True}
