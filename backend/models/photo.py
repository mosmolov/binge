from beanie import Document
from pydantic import BaseModel
from typing import Optional, List

class Photo(Document):
    _id: str
    photo_id: str
    business_id: str
    caption: Optional[str]
    label: Optional[str]
    class Settings:
        name = "photos"  # Collection name in MongoDB
        