from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from backend.database import init_db
from backend.routers.photos import router as photos_router
from backend.routers.restaurants import router as restaurants_router
@asynccontextmanager
async def get_db(app: FastAPI):
    # Call init_db() without unpacking
    db = await init_db()
    try:
        yield db
    finally:
        pass

app = FastAPI(title="Binge API", lifespan=get_db)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(photos_router)
app.include_router(restaurants_router)

@app.get("/")
async def root():
    return {"message": "Welcome to Binge API"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)