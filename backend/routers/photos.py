from beanie import init_beanie
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from backend.models.photo import Photo

router = APIRouter()
@router.get("/photos", response_model=List[Photo])
async def get_photos():
    photos = await Photo.all().to_list()
    return photos

