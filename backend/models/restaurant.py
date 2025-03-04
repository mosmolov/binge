from beanie import Document

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