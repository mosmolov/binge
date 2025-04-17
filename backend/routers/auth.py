from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from beanie import PydanticObjectId
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
import os
from dotenv import load_dotenv
from backend.models.user import User

load_dotenv()

router = APIRouter(prefix="/auth", tags=["Auth"])

# Security settings
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")  # replace or set in .env
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Pydantic models for request/response
class UserCreate(BaseModel):
    username: str
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# Route to register a new user
@router.post("/register", status_code=201)
async def register(user: UserCreate):
    # Check if username or email already exists
    existing = await User.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already registered")

    hashed_pwd = get_password_hash(user.password)
    new_user = User(username=user.username, email=user.email, password=hashed_pwd)
    await new_user.insert()
    return {"id": str(new_user.id), "username": new_user.username, "email": new_user.email}

# Route to login and get JWT token
token_model = Token
@router.post("/login", response_model=token_model)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    # Authenticate user by username
    user = await User.find_one(User.username == form_data.username)
    if not user or not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

# Dependency to get current user
async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await User.get(PydanticObjectId(user_id))
    if user is None:
        raise credentials_exception
    return user

# Route to get current user info
@router.get("/me")
async def read_users_me(current_user: User = Depends(get_current_user)):
    return {"id": str(current_user.id), "username": current_user.username, "email": current_user.email}