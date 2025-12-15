from pydantic import EmailStr, Field
from typing import List, Optional
from app.models.base import MongoBaseModel, PyObjectId

class Observer(MongoBaseModel):
    name: str = Field(..., min_length=3)
    email: EmailStr = Field(...)
    password: str = Field(..., min_length=6)
    phoneNumber: Optional[str] = None
    patient_ids: List[PyObjectId] = Field(default_factory=list)

class ObserverCreate(Observer):
    class Config:
        exclude = {'id'}

class ObserverLogin(MongoBaseModel):
    email: EmailStr
    password: str