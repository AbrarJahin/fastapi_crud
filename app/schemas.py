from pydantic import BaseModel, Field, ConfigDict


class ItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None


class ItemUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None


class ItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None = None


class PaginatedItems(BaseModel):
    total: int
    skip: int
    limit: int
    items: list[ItemOut]
