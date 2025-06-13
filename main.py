from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sympy as sp

# Intentamos importar el parser LaTeX
try:
    from sympy.parsing.latex import parse_latex
    _can_parse_latex = True
except ImportError:
    _can_parse_latex = False

app = FastAPI()

class RequestData(BaseModel):
    equation: str   | None = None  # cadena LaTeX o texto
    a:        float | None = None  # coeficiente
    altura:   float | None = None  # altura H
    ancho:    float             # ancho W

class ResponseData(BaseModel):
    equation:   str
    area_total: float
    points:     list[tuple[float,float,float]]

@app.post("/calcular", response_model=ResponseData)
def calcular(data: RequestData):
    try:
        # Chequeo ancho
        W = data.ancho

        # Variable simbólica
        x, y = sp.symbols('x y')

        # ── MODO ECUACIÓN ────────────────────────────────
        if data.equation:
            raw = data.equation.strip()

            # 1) Si hay un "=", lo separamos en dos lados
            if '=' in raw:
                left, right = raw.split('=', 1)

                # Parseo de cada lado
                if _can_parse_latex and '\\' in raw:
                    L = parse_latex(left)
                    R = parse_latex(right)
                else:
                    L = sp.sympify(left.replace('^','**'))
                    R = sp.sympify(right.replace('^','**'))

                # Ecuación implícita L - R = 0
                eq0 = L - R

                # Despejamos y → obtenemos lista de soluciones
                sols = sp.solve(eq0, y)
                if not sols:
                    raise ValueError("No pude despejar 'y' de la ecuación")
                expr = sols[0]  # primera rama

                eq_text = f"y(x) = {sp.sstr(expr)}"

            else:
                # No hay "=", asumimos función explícita
                if _can_parse_latex and '\\' in raw:
                    expr = parse_latex(raw)
                else:
                    expr = sp.sympify(raw.replace('^','**'))
                eq_text = f"y(x) = {raw}"

        # ── MODO PARÁMETRO ────────────────────────────────
        else:
            # Priorizar coeficiente 'a'
            if data.a is not None:
                a = data.a
                H = 0.0
            elif data.altura is not None:
                H = data.altura
                a = -4 * H / (W**2)
            else:
                raise ValueError("Debes enviar 'equation', o bien 'a' o 'altura'")

            expr    = a * x**2 + H
            eq_text = f"y(x) = {a:.4f}·x² + {H:.4f}"

        # ── CÁLCULO DE ÁREA Y PUNTOS ───────────────────────
        area = float(sp.integrate(expr, (x, -W/2, W/2)))
        xs   = np.linspace(-W/2, W/2, 50)
        pts  = [(float(xi), float(expr.subs(x, xi)), 0.0) for xi in xs]

        return {
            "equation":   eq_text,
            "area_total": round(area, 2),
            "points":     pts
        }

    except Exception as e:
        # Devolveremos 400 con detalle del error
        raise HTTPException(status_code=400, detail=str(e))
