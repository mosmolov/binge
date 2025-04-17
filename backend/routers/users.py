from fastapi import APIRouter, HTTPException, Depends, status, Query
from typing import List, Optional
from beanie import PydanticObjectId

from backend.models.user import User

router = APIRouter(prefix="/users", tags=["users"])

# Create a new user
@router.post("/", response_model=User)
async def create_user(user: User):
    try:
        created = await user.insert()
        return created
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

# Get user by ID
@router.get("/{user_id}", response_model=User)
async def get_user(user_id: PydanticObjectId):
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return user

# Add a liked business ID to user preferences
@router.patch("/{user_id}/likes", response_model=User)
async def add_user_like(user_id: PydanticObjectId, business_id: str = Query(..., description="Business ID to like")):
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if business_id not in user.liked_business_ids:
        user.liked_business_ids.append(business_id)
        await user.save()
    return user

# Add a disliked business ID to user preferences
@router.patch("/{user_id}/dislikes", response_model=User)
async def add_user_dislike(user_id: PydanticObjectId, business_id: str = Query(..., description="Business ID to dislike")):
    user = await User.get(user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if business_id not in user.disliked_business_ids:
        user.disliked_business_ids.append(business_id)
        await user.save()
    return user