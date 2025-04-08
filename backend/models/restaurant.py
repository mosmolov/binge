from beanie import Document
from pydantic import BaseModel
from typing import Optional, List
class Restaurant(Document):
    _id: str
    # string or int
    business_id: str | int
    name: str
    address: str
    latitude: float
    longitude: float
    stars: float
    
    class Settings:
        name = "restaurants"