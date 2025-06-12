from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from sympy import symbols, integrate
from sympy.parsing.latex import parse_latex

app = FastAPI()

class RequestData(BaseModel):
    equation: str   | None = None
    a:        float | None = None
    altura:   float | None = None
    ancho:    float

class ResponseData(BaseModel):
    equation:   str
    area_total: float
    points:     list[tuple[float,float,float]]

@app.post("/calcular", response_model=ResponseData)
def calcular(data: RequestData):
    try:
        if data.ancho is None:
            raise ValueError("'ancho' es obligatorio")
        W = data.ancho
        x = symbols('x')

        # 1) Modo LaTeX
        if data.equation:
            expr    = parse_latex(data.equation)
            eq_text = f"y(x) = {data.equation}"

        # 2) Modo parámetro: 'a' o derivar desde 'altura'
        else:
            if data.a is not None:
                a = data.a
            elif data.altura is not None:
                a = -4 * data.altura / (W**2)
            else:
                raise ValueError("Debes enviar 'equation' o el coeficiente 'a' o la 'altura'")
            expr    = a * x**2
            eq_text = f"y(x) = {a:.4f}·x²"

        # Cálculo de área y puntos
        area = float(integrate(expr, (x, -W/2, W/2)))
        xs   = np.linspace(-W/2, W/2, 50)
        pts  = [(float(xi), float(expr.subs(x, xi)), 0.0) for xi in xs]

        return {
            "equation":   eq_text,
            "area_total": round(area, 2),
            "points":     pts
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
