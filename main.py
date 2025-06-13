import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import sympy as sp

# Para LaTeX
try:
    from sympy.parsing.latex import parse_latex
    _can_parse_latex = True
except ImportError:
    _can_parse_latex = False

# Para texto puro con multiplicación implícita
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

_transformations = standard_transformations + (
    implicit_multiplication_application,
)

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
        W = data.ancho
        x, y = sp.symbols('x y')

        if data.equation:
            raw = data.equation.strip()

            # Si hay “=”, lo resolvemos implícito
            if '=' in raw:
                left, right = raw.split('=', 1)

                def _prepare(s: str) -> str:
                    # 1) caret -> **
                    t = s.replace('^','**')
                    # 2) inyectar "*" entre dígito y letra: 12x -> 12*x, y2 -> y*2
                    t = re.sub(r'(?<=\d)(?=[A-Za-z])', '*', t)
                    t = re.sub(r'(?<=[A-Za-z])(?=\d)', '*', t)
                    return t

                if _can_parse_latex and '\\' in raw:
                    L = parse_latex(left)
                    R = parse_latex(right)
                else:
                    L = parse_expr(_prepare(left),
                                   transformations=_transformations)
                    R = parse_expr(_prepare(right),
                                   transformations=_transformations)

                eq0 = L - R
                sols = sp.solve(eq0, y)
                if not sols:
                    raise ValueError("No pude despejar 'y'")
                expr = sols[0]
                eq_text = f"y(x) = {sp.sstr(expr)}"

            else:
                # función directa, sin “=”
                if _can_parse_latex and '\\' in raw:
                    expr = parse_latex(raw)
                else:
                    prepared = re.sub(r'(?<=\d)(?=[A-Za-z])', '*',
                              raw.replace('^','**'))
                    prepared = re.sub(r'(?<=[A-Za-z])(?=\d)', '*', prepared)
                    expr = parse_expr(prepared,
                                      transformations=_transformations)
                eq_text = f"y(x) = {raw}"

        else:
            # Modo parámetro
            if data.a is not None:
                a, H = data.a, 0.0
            elif data.altura is not None:
                H = data.altura
                a = -4 * H / (W**2)
            else:
                raise ValueError("Envía 'equation', o bien 'a' o 'altura'")
            expr = a * x**2 + H
            eq_text = f"y(x) = {a:.4f}·x² + {H:.4f}"

        # Integramos y generamos puntos
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
