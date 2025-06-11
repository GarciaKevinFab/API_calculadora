from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sympy as sp

from calculadora import expr_parabola_canonica, integra_area, genera_puntos

app = FastAPI()

class RequestData(BaseModel):
    altura:   float = None   # H (solo en modo medidas)
    ancho:    float = None   # W (siempre obligatorio)
    equation: str   = None   # f(x) estilo GeoGebra

class ResponseData(BaseModel):
    equation:   str
    area_total: float
    points:     list[tuple[float,float,float]]

@app.post("/calcular", response_model=ResponseData)
def calcular(data: RequestData):
    try:
        if data.ancho is None:
            raise ValueError("El campo 'ancho' es obligatorio")
        W = data.ancho

        if data.equation:
            raw = data.equation.replace('^', '**')
            x = sp.symbols('x')
            expr = sp.sympify(raw)
            eq_text = f"y(x) = {raw}"
        else:
            if data.altura is None:
                raise ValueError("Debes enviar 'altura' si no usas 'equation'")
            H = data.altura
            expr, eq_text = expr_parabola_canonica(H, W)

        area = integra_area(expr, W)
        pts  = genera_puntos(expr, W, num=50)
        return {
            "equation": eq_text,
            "area_total": round(area, 2),
            "points": pts
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
