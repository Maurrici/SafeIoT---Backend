from typing import List
from fastapi import APIRouter, HTTPException, Query, status
from app.models.patient import Patient, PatientCreate, PatientLogin
from app.database import patient_collection
from bson import ObjectId

router = APIRouter(
    prefix="/patients",
    tags=["Patients"]
)

@router.post("/", response_model=Patient, status_code=status.HTTP_201_CREATED)
async def create_patient(patient: PatientCreate):
    patient_dict = patient.model_dump(by_alias=True)
    
    if "_id" in patient_dict:
        del patient_dict["_id"]
    
    result = await patient_collection.insert_one(patient_dict)
    new_patient = await patient_collection.find_one({"_id": result.inserted_id})
    
    if new_patient is None:
         raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve inserted patient."
        )

    return Patient(**new_patient)

@router.post("/login", response_model=Patient, response_model_exclude={"password"})
async def login_patient(credentials: PatientLogin):

    patient = await patient_collection.find_one({"email": credentials.email})

    if not patient or patient.get("password") != credentials.password:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ou Senha incorretos."
        )
    return Patient(**patient)

@router.get("/", response_model=List[Patient])
async def list_patients_by_observer(
    observer_id: str = Query(
        ..., 
        description="ID do observador para filtrar os pacientes observados. (Obrigatório)"
    )
):
    """
    Retorna os pacientes observados pelo Observador especificado.
    """
    
    if not ObjectId.is_valid(observer_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Formato de ID de Observador inválido."
        )
    
    observer_oid = ObjectId(observer_id)
    query = {"observer_ids": observer_oid}
    patients = []

    async for patient_doc in patient_collection.find(query).limit(100):
        patients.append(Patient(**patient_doc))
        
    if not patients:
        pass
        
    return patients

@router.get("/{id}", response_model=Patient)
async def get_patient(id: str):
    """Retorna um paciente pelo ID."""
    if not ObjectId.is_valid(id):
        raise HTTPException(status_code=400, detail="Invalid ID format")

    patient = await patient_collection.find_one({"_id": ObjectId(id)})
    if patient is None:
        raise HTTPException(status_code=404, detail="Patient not found")
    
    return Patient(**patient)

