from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sympy import symbols, integrate
from sympy.parsing.latex import parse_latex

from calculadora import expr_parabola_canonica, genera_puntos  # tu módulo de helpers

app = FastAPI()

class RequestData(BaseModel):
    altura:   float | None = None
    ancho:    float
    equation: str   | None = None

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
        x = symbols('x')

        # 1) Ecuación LaTeX
        if data.equation:
            # parse_latex maneja \frac, exponentes, raíces, etc.
            expr = parse_latex(data.equation)
            eq_text = f"y(x) = {data.equation}"

        # 2) Parábola canónica con altura+ancho
        else:
            if data.altura is None:
                raise ValueError("Debes enviar 'altura' si no usas 'equation'")
            H = data.altura
            expr, eq_text = expr_parabola_canonica(H, W)

        # 3) Integración y puntos
        area = float(integrate(expr, (x, -W/2, W/2)))
        pts  = genera_puntos(expr, W, num=50)

        return {
            "equation":   eq_text,
            "area_total": round(area, 2),
            "points":     pts
        }

    except Exception as e:
        # Devuelve 400 con el mensaje de error de Sympy o tus validaciones
        raise HTTPException(status_code=400, detail=str(e))
