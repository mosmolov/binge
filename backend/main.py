from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
from contextlib import asynccontextmanager
from logging import info
from beanie import init_beanie
from backend.routers import photos
from backend.models import photo

CONNECTION_STRING = os.getenv("MONGO_URI")
DATABASE_NAME = os.getenv("DATABASE_NAME", "default_database_name")  # Replace with your default database name

@asynccontextmanager
async def db_lifespan(app: FastAPI):
    # Startup
    app.mongodb_client = AsyncIOMotorClient(CONNECTION_STRING)
    app.database = app.mongodb_client[DATABASE_NAME]
    await init_beanie(database=app.database, document_models=[photo.Photo])
    ping_response = await app.database.command("ping")
    if int(ping_response["ok"]) != 1:
        raise Exception("Problem connecting to database cluster.")
    else:
        info("Connected to database cluster.")
    
    yield
    # Shutdown
    app.mongodb_client.close()

app = FastAPI(lifespan=db_lifespan)

app.include_router(photos.router)