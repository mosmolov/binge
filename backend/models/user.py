from beanie import Document

from pydantic import BaseModel, Field
from typing import Optional, List


class User(Document):
    username: str
    email: str
    password: str
    liked_business_ids: List[str] = Field(default_factory=list)
    disliked_business_ids: List[str] = Field(default_factory=list)
    visited_business_ids: List[str] = Field(default_factory=list)    

    class Settings:
        name = "users"