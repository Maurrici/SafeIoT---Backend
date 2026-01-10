from fastapi import APIRouter, HTTPException, Query, status, BackgroundTasks
from typing import List
from bson import ObjectId
from pydantic import BaseModel

from app.models.observer import Observer, ObserverCreate, ObserverLogin
from app.models.patient import Patient
from app.database import observer_collection, patient_collection

from app.routes.observer_notifications import notify_observer_onboarding

router = APIRouter(prefix="/observers", tags=["Observers"])


# ---------------------------------------------------------------------------
# Schemas auxiliares
# ---------------------------------------------------------------------------
class TelegramRegister(BaseModel):
    telegram_chat_id: int


# ---------------------------------------------------------------------------
# CREATE OBSERVER
# ---------------------------------------------------------------------------
@router.post("/", response_model=Observer, status_code=status.HTTP_201_CREATED)
async def create_observer(
    observer: ObserverCreate,
    background_tasks: BackgroundTasks
):
    observer_dict = observer.model_dump(by_alias=True)

    if await observer_collection.find_one({"email": observer_dict["email"]}):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Observer with this email already exists"
        )

    observer_dict.pop("_id", None)

    result = await observer_collection.insert_one(observer_dict)
    observer_db = await observer_collection.find_one({"_id": result.inserted_id})

    # Onboarding por e-mail
    await notify_observer_onboarding(observer_db, background_tasks)

    return Observer(**observer_db)


# ---------------------------------------------------------------------------
# LOGIN
# ---------------------------------------------------------------------------
@router.post("/login", response_model=Observer, response_model_exclude={"password"})
async def login_observer(
    credentials: ObserverLogin,
    background_tasks: BackgroundTasks
):
    observer = await observer_collection.find_one({"email": credentials.email})

    if not observer or observer.get("password") != credentials.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou Senha incorretos."
        )

    # Se ainda não tem Telegram, reenvia onboarding
    if observer.get("telegram_chat_id") is None:
        await notify_observer_onboarding(observer, background_tasks)

    return Observer(**observer)


# ---------------------------------------------------------------------------
# LIST OBSERVERS BY PATIENT  (CRÍTICA PARA O BOT)
# ---------------------------------------------------------------------------
@router.get("/", response_model=List[Observer])
async def list_observers_by_patient(
    patient_id: str = Query(..., description="ID do paciente")
):
    if not ObjectId.is_valid(patient_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato de ID inválido"
        )

    pid = ObjectId(patient_id)
    observers = []

    async for doc in observer_collection.find({"patient_ids": pid}):
        observers.append(Observer(**doc))

    return observers


# ---------------------------------------------------------------------------
# SEARCH BY EMAIL
# ---------------------------------------------------------------------------
@router.get("/search", status_code=status.HTTP_200_OK)
async def search_user_by_email(email: str, target_type: str):
    collection = patient_collection if target_type == "patient" else observer_collection
    user = await collection.find_one({"email": email})

    if not user:
        raise HTTPException(status_code=404, detail="Usuário não encontrado")

    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"]
    }


# ---------------------------------------------------------------------------
# GET OBSERVER BY ID
# ---------------------------------------------------------------------------
@router.get("/{id}", response_model=Observer)
async def get_observer(id: str):
    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="Invalid Observer ID")

    observer = await observer_collection.find_one({"_id": ObjectId(id)})

    if not observer:
        raise HTTPException(status_code=404, detail="Observer not found")

    return Observer(**observer)


# ---------------------------------------------------------------------------
# LINK PATIENT <-> OBSERVER
# ---------------------------------------------------------------------------
@router.post("/{observer_id}/patients/{patient_id}", response_model=Observer)
async def link_patient_to_observer(observer_id: str, patient_id: str):
    if not ObjectId.is_valid(observer_id) or not ObjectId.is_valid(patient_id):
        raise HTTPException(status_code=400, detail="Invalid ID")

    oid = ObjectId(observer_id)
    pid = ObjectId(patient_id)

    observer = await observer_collection.find_one({"_id": oid})
    patient = await patient_collection.find_one({"_id": pid})

    if not observer or not patient:
        raise HTTPException(status_code=404, detail="Observer or Patient not found")

    await observer_collection.update_one(
        {"_id": oid},
        {"$addToSet": {"patient_ids": pid}}
    )

    await patient_collection.update_one(
        {"_id": pid},
        {"$addToSet": {"observer_ids": oid}}
    )

    updated = await observer_collection.find_one({"_id": oid})
    return Observer(**updated)


# ---------------------------------------------------------------------------
# REGISTER TELEGRAM CHAT ID
# ---------------------------------------------------------------------------
@router.post("/{observer_id}/telegram", response_model=Observer)
async def register_telegram_chat_id(observer_id: str, payload: TelegramRegister):
    if not ObjectId.is_valid(observer_id):
        raise HTTPException(status_code=400, detail="Invalid Observer ID")

    oid = ObjectId(observer_id)

    await observer_collection.update_one(
        {"_id": oid},
        {"$set": {"telegram_chat_id": payload.telegram_chat_id}}
    )

    observer = await observer_collection.find_one({"_id": oid})
    return Observer(**observer)
