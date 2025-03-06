from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
from beanie.operators import In, RegEx
from backend.models.photo import Photo

router = APIRouter(prefix="/photos", tags=["photos"])

@router.get("/", response_model=List[Photo], tags=["photos"])
async def get_photos():
    try:
        photos = await Photo.find_all().to_list(length=100)
        return photos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/random", response_model=List[Photo], tags=["photos"])
async def get_random_photos(limit: int = 20):
    try:
        pipeline = [{"$sample": {"size": limit}}]
        random_photos = await Photo.aggregate(pipeline).to_list()
        return random_photos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{photo_id}", response_model=Photo)
async def get_photo(photo_id: str):
    try:
        photo = await Photo.get(photo_id)
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        return photo
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/{business_id}", response_model=List[Photo])
async def get_business_photos(business_id: str):
    try:
        photos = await Photo.find_all({"business_id": business_id}).to_list()
        return photos
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.post("/", response_model=Photo, status_code=status.HTTP_201_CREATED)
async def create_photo(photo: Photo):
    try:
        await photo.create()
        return photo
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{photo_id}", response_model=Photo)
async def update_photo(photo_id: str, photo_data: Photo):
    try:
        photo = await Photo.get(photo_id)
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        
        # Update all fields excluding ID
        update_dict = photo_data.model_dump(exclude={"id"})
        for field, value in update_dict.items():
            setattr(photo, field, value)
            
        await photo.save()
        return photo
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{photo_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_photo(photo_id: str):
    try:
        photo = await Photo.get(photo_id)
        if not photo:
            raise HTTPException(status_code=404, detail="Photo not found")
        await photo.delete()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))