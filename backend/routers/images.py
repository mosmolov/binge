from fastapi import APIRouter, HTTPException, status
from typing import List, Optional
import cloudinary
from cloudinary.utils import cloudinary_url
import os
import dotenv
dotenv.load_dotenv()
cloudinary.config(
    cloud_name="diub0blpa",
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

router = APIRouter(prefix="/images", tags=["images"])

@router.get("/", response_model=dict, tags=["images"])
async def get_images():
    try:
        images = cloudinary.Search().expression("asset_folder:binge_photos").sort_by("public_id", "desc").max_results(5).execute()
        return images
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{image_id}", response_model=dict, tags=["images"])
async def get_image(image_id: str):
    try:
        image = cloudinary.Search().expression(f"display_name:{image_id}.jpg").execute()
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))