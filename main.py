from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sympy as sp

from calculadora import expr_parabola_canonica, integra_area, genera_puntos

app = FastAPI()

class RequestData(BaseModel):
    # Modo medidas (canónica)
    altura: float = None
    ancho:  float = None
    # Modo ecuación libre (any f(x))
    equation: str = None  # p.ej. "0.5*x^2 + sin(x)"

class ResponseData(BaseModel):
    equation: str               # la ecuación usada, como texto
    area_total: float
    points: list[tuple[float,float,float]]

@app.post("/calcular", response_model=ResponseData)
def calcular(data: RequestData):
    try:
        # ancho es obligatorio en ambos modos
        if data.ancho is None:
            raise ValueError("El campo 'ancho' es obligatorio")

        W = data.ancho

        # 1) Elegir modo: equation libre o canónica por medidas
        if data.equation:
            # parsing: ^ → **
            raw = data.equation.replace('^', '**')
            x = sp.symbols('x')
            expr = sp.sympify(raw)
            eq_text = f"y(x) = {raw}"
        else:
            # parábola canónica de altura H sobre ancho W
            if data.altura is None:
                raise ValueError("Debes enviar 'altura' si no usas 'equation'")
            H = data.altura
            expr, eq_text = expr_parabola_canonica(H, W)

        # 2) Cálculo del área bajo y(x) entre -W/2 y +W/2
        area = integra_area(expr, W)

        # 3) Generación de puntos 3D para graficar
        pts = genera_puntos(expr, W, num=50)

        return {
            "equation": eq_text,
            "area_total": round(area, 2),
            "points": pts
        }

    except Exception as e:
        # devolvemos 400 con el mensaje de error
        raise HTTPException(status_code=400, detail=str(e))