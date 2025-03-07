from beanie import Document
from pydantic import BaseModel
from typing import Optional, List
class Restaurant(Document):
    _id: str
    business_id: str
    name: str
    address: str
    city: str
    state: str
    postal_code: str
    latitude: float
    longitude: float
    stars: float
    
    class Settings:
        name = "restaurants"