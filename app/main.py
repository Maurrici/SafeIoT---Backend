from fastapi import FastAPI
from app.routes import patient as patient_routes
from app.routes import observer as observer_routes
from app.routes import fall_register as fall_register_routes

app = FastAPI(
    title="API de Detecção de Queda de Idosos",
    description="API para gerenciar pacientes, observadores e registros de queda.",
    version="1.0.0",
)

app.include_router(patient_routes.router)
app.include_router(observer_routes.router)
app.include_router(fall_register_routes.router)

@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "API de Detecção de Queda funcionando. Acesse /docs para a documentação Swagger."}