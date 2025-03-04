from beanie import Document

class Photo(Document):
    _id: str
    photo_id: str
    business_id: str
    caption: str
    label: str
    url: str
    title: str

    class Settings:
        name = "photos"