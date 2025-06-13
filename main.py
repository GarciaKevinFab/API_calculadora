from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sympy as sp

# Intentamos importar parse_latex; si falla, caemos en fallback
try:
    from sympy.parsing.latex import parse_latex
    _can_parse_latex = True
except ImportError:
    _can_parse_latex = False

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
        x = sp.symbols('x')

        # ── MODO ECUACIÓN (LaTeX o texto) ───────────────────────
        if data.equation:
            raw_eq = data.equation.strip()

            if _can_parse_latex and '\\' in raw_eq:
                # Si hay comandos LaTeX y el runtime está disponible
                expr = parse_latex(raw_eq)
            else:
                # Fallback: texto puro, convertimos ^→** y sympify
                raw = raw_eq.replace('^', '**')
                expr = sp.sympify(raw)

            eq_text = f"y(x) = {raw_eq}"

        # ── MODO PARÁMETRO ───────────────────────────────────────
        else:
            # Preferimos coeficiente 'a' si se proporciona
            if data.a is not None:
                a = data.a
                H = 0.0
            elif data.altura is not None:
                H = data.altura
                a = -4 * H / (W**2)
            else:
                raise ValueError("Debes enviar 'equation' o 'a' o 'altura'")

            expr    = a * x**2 + H
            eq_text = f"y(x) = {a:.4f}·x² + {H:.4f}"

        # ── CÁLCULO DE ÁREA Y PUNTOS ───────────────────────────
        area = float(sp.integrate(expr, (x, -W/2, W/2)))
        xs   = np.linspace(-W/2, W/2, 50)
        pts  = [(float(xi), float(expr.subs(x, xi)), 0.0) for xi in xs]

        return {
            "equation":   eq_text,
            "area_total": round(area, 2),
            "points":     pts
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
