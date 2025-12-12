from fastapi import APIRouter, HTTPException, Query, status
from typing import List
from bson import ObjectId

from app.models.observer import Observer, ObserverCreate
from app.models.patient import Patient
from app.database import observer_collection, patient_collection

router = APIRouter(
    prefix="/observers",
    tags=["Observers"]
)

@router.post("/", response_model=Observer, status_code=status.HTTP_201_CREATED)
async def create_observer(observer: ObserverCreate):
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
    
    return Observer(**new_observer)

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

@router.get("/{id}", response_model=Observer)
async def get_observer(id: str):
    """Retorna um observador pelo ID."""
    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="Invalid Observer ID format")

    observer = await observer_collection.find_one({"_id": ObjectId(id)})
    if observer is None:
        raise HTTPException(status_code=404, detail="Observer not found")
    
    return Observer(**observer)

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
        raise HTTPException(status_code=404, detail=f"Observer with ID {observer_id} not found.")
    
    if patient_doc is None:
        raise HTTPException(status_code=404, detail=f"Patient with ID {patient_id} not found.")
    
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