from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from api.calculadora import calcular_area_muro

app = FastAPI()

class RequestData(BaseModel):
    altura: float
    ancho: float
    a: float = None  # opcional

class ResponseData(BaseModel):
    area_total: float

@app.get("/")
async def read_root():
    return {"message": "¡API alive and kicking!"}

@app.post("/calcular", response_model=ResponseData)
def calcular(data: RequestData):
    try:
        area = calcular_area_muro(data.altura, data.ancho, data.a)
        return {"area_total": round(area, 2)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))