from beanie import Document
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class Restaurant(Document):
    _id: str
    # string or int
    business_id: str | int
    name: str
    address: str
    latitude: float
    longitude: float
    stars: float
    price: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = None
    
    class Settings:
        name = "restaurants"
        
class RestaurantCreate(BaseModel):
    name: str
    address: str
    latitude: float
    longitude: float
    stars: float
    price: Optional[int] = None
    attributes: Optional[Dict[str, Any]] = None