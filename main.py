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

def normalize_superscripts(s: str) -> str:
    """Convierte dígitos superíndice Unicode a ^n."""
    return (
        s.replace('⁰', '^0')
         .replace('¹', '^1')
         .replace('²', '^2')
         .replace('³', '^3')
         .replace('⁴', '^4')
         .replace('⁵', '^5')
         .replace('⁶', '^6')
         .replace('⁷', '^7')
         .replace('⁸', '^8')
         .replace('⁹', '^9')
    )

def prepare_expr(s: str) -> str:
    """
    1) Normaliza superscripts.
    2) '^'→'**'
    3) Inserta '*' entre número-letra y letra-número (12x→12*x, y2→y*2).
    """
    t = normalize_superscripts(s)
    t = t.replace('^', '**')
    t = re.sub(r'(?<=\d)(?=[A-Za-z])', '*', t)
    t = re.sub(r'(?<=[A-Za-z])(?=\d)', '*', t)
    return t

@app.post("/calcular", response_model=ResponseData)
def calcular(data: RequestData):
    try:
        W = data.ancho
        x, y = sp.symbols('x y')

        if data.equation:
            raw = data.equation.strip()

            # ── Ecuación implícita con '=' ─────────────────────────
            if '=' in raw:
                left, right = raw.split('=', 1)

                if _can_parse_latex and '\\' in raw:
                    L = parse_latex(left)
                    R = parse_latex(right)
                else:
                    L = parse_expr(prepare_expr(left),
                                   transformations=_transformations)
                    R = parse_expr(prepare_expr(right),
                                   transformations=_transformations)

                eq0 = L - R
                sols = sp.solve(eq0, y)
                if not sols:
                    raise ValueError("No pude despejar 'y'")
                expr = sols[0]
                eq_text = f"y(x) = {sp.sstr(expr)}"

            # ── Función explícita sin '=' ──────────────────────────
            else:
                if _can_parse_latex and '\\' in raw:
                    expr = parse_latex(raw)
                else:
                    expr = parse_expr(prepare_expr(raw),
                                      transformations=_transformations)
                eq_text = f"y(x) = {raw}"

        else:
            # ── Modo Parámetro ───────────────────────────────────
            if data.a is not None:
                a, H = data.a, 0.0
            elif data.altura is not None:
                H = data.altura
                a = -4 * H / (W**2)
            else:
                raise ValueError("Envía 'equation', o bien 'a' o 'altura'")

            expr    = a * x**2 + H
            eq_text = f"y(x) = {a:.4f}·x² + {H:.4f}"

        # ── Cálculo de Área y Puntos ───────────────────────────
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
