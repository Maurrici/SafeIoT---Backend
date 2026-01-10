from fastapi import APIRouter, HTTPException, Query, status, BackgroundTasks
from typing import List
from bson import ObjectId
from pydantic import BaseModel

from app.models.observer import Observer, ObserverCreate, ObserverLogin
from app.models.patient import Patient
from app.database import observer_collection, patient_collection

# >>> NOVO (opcional): onboarding Telegram / e-mail
from app.routes.observer_notifications import notify_observer_onboarding
# <<<

router = APIRouter(
    prefix="/observers",
    tags=["Observers"]
)

# ============================================================================
# SCHEMAS AUXILIARES (NOVO)
# ============================================================================
class TelegramRegister(BaseModel):
    telegram_chat_id: int


# ============================================================================
# CREATE OBSERVER  (INALTERADO + ONBOARDING)
# ============================================================================
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

    if "_id" in observer_dict:
        del observer_dict["_id"]

    result = await observer_collection.insert_one(observer_dict)
    new_observer = await observer_collection.find_one({"_id": result.inserted_id})

    if new_observer is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve inserted observer."
        )

    # >>> NOVO: onboarding (Telegram / e-mail)
    await notify_observer_onboarding(new_observer, background_tasks)
    # <<<

    return Observer(**new_observer)


# ============================================================================
# LOGIN  (INALTERADO + FALLBACK ONBOARDING)
# ============================================================================
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

    # >>> NOVO: se ainda não tem Telegram, reenvia onboarding
    if observer.get("telegram_chat_id") is None:
        await notify_observer_onboarding(observer, background_tasks)
    # <<<

    return Observer(**observer)


# ============================================================================
# LIST OBSERVERS BY PATIENT  (INALTERADO – CRÍTICO PARA O BOT)
# ============================================================================
@router.get("/", response_model=List[Observer])
async def list_observers_by_patient(
    patient_id: str = Query(
        ...,
        description="ID do paciente para filtrar os observadores que o acompanham. (Obrigatório)"
    )
):
    """
    Retorna os observadores que estão vinculados ao Paciente especificado.
    """
    if not ObjectId.is_valid(patient_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Formato de ID de Paciente inválido."
        )

    pid = ObjectId(patient_id)
    query = {"patient_ids": pid}
    observers = []

    async for observer_doc in observer_collection.find(query).limit(100):
        observers.append(Observer(**observer_doc))

    return observers


# ============================================================================
# SEARCH USER BY EMAIL  (INALTERADO)
# ============================================================================
@router.get("/search", status_code=status.HTTP_200_OK)
async def search_user_by_email(email: str, target_type: str):
    """
    Busca um usuário pelo e-mail.
    target_type deve ser 'patient' ou 'observer'.
    """
    collection = patient_collection if target_type == "patient" else observer_collection
    user = await collection.find_one({"email": email})

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{target_type.capitalize()} not found with email {email}"
        )

    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"]
    }


# ============================================================================
# GET OBSERVER BY ID  (INALTERADO)
# ============================================================================
@router.get("/{id}", response_model=Observer)
async def get_observer(id: str):
    """Retorna um observador pelo ID."""
    if not ObjectId.is_valid(id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Observer ID format"
        )

    observer = await observer_collection.find_one({"_id": ObjectId(id)})
    if observer is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Observer not found"
        )

    return Observer(**observer)


# ============================================================================
# LINK PATIENT <-> OBSERVER  (INALTERADO)
# ============================================================================
@router.post("/{observer_id}/patients/{patient_id}", response_model=Observer)
async def link_patient_to_observer(observer_id: str, patient_id: str):
    """
    Vincula um Paciente a um Observador.
    Atualiza ambos os documentos no MongoDB para registrar o relacionamento.
    """
    if not ObjectId.is_valid(observer_id) or not ObjectId.is_valid(patient_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format for Observer or Patient."
        )

    oid = ObjectId(observer_id)
    pid = ObjectId(patient_id)

    observer_doc = await observer_collection.find_one({"_id": oid})
    patient_doc = await patient_collection.find_one({"_id": pid})

    if observer_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Observer with ID {observer_id} not found."
        )

    if patient_doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Patient with ID {patient_id} not found."
        )

    await observer_collection.update_one(
        {"_id": oid},
        {"$addToSet": {"patient_ids": pid}}
    )

    await patient_collection.update_one(
        {"_id": pid},
        {"$addToSet": {"observer_ids": oid}}
    )

    updated_observer = await observer_collection.find_one({"_id": oid})
    return Observer(**updated_observer)


# ============================================================================
# REGISTER TELEGRAM CHAT ID  (NOVO – ISOLADO)
# ============================================================================
@router.post("/{observer_id}/telegram", response_model=Observer)
async def register_telegram_chat_id(observer_id: str, payload: TelegramRegister):
    """
    Registra ou atualiza o chat_id do Telegram para um observador.
    """
    if not ObjectId.is_valid(observer_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid Observer ID format"
        )

    oid = ObjectId(observer_id)

    await observer_collection.update_one(
        {"_id": oid},
        {"$set": {"telegram_chat_id": payload.telegram_chat_id}}
    )

    observer = await observer_collection.find_one({"_id": oid})
    return Observer(**observer)
