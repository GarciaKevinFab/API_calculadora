import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import sympy as sp

# Intentar usar parser de LaTeX
try:
    from sympy.parsing.latex import parse_latex
    _can_parse_latex = True
except ImportError:
    _can_parse_latex = False

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
_transformations = standard_transformations + (implicit_multiplication_application,)

app = FastAPI()

# ─── Habilitar CORS ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # O restringe a tus dominios
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ─────────────────────────────────────────────────────────────

class RequestData(BaseModel):
    equation: str | None = None
    a: float | None = None
    altura: float | None = None
    ancho: float

class ResponseData(BaseModel):
    equation: str
    area_total: float
    points: list[tuple[float, float, float]]
    vertex: tuple[float, float, float]
    focus: tuple[float, float, float]

def normalize_superscripts(s: str) -> str:
    return (s.replace('⁰','^0').replace('¹','^1')
             .replace('²','^2').replace('³','^3')
             .replace('⁴','^4').replace('⁵','^5')
             .replace('⁶','^6').replace('⁷','^7')
             .replace('⁸','^8').replace('⁹','^9'))

def prepare_expr(s: str) -> str:
    t = normalize_superscripts(s)
    t = t.replace('^','**')
    t = re.sub(r'(?<=\d)(?=[A-Za-z])','*',t)
    t = re.sub(r'(?<=[A-Za-z])(?=\d)','*',t)
    t = re.sub(r"y'","Derivative(y, x)",t)
    return t

@app.post("/calcular", response_model=ResponseData)
def calcular(data: RequestData):
    try:
        W = data.ancho
        x, y = sp.symbols('x y')

        # ── 1) Construir la expresión y(x)
        if data.equation:
            raw   = data.equation.strip()
            final = raw.split('=', 1)[-1].strip()  # solo RHS
            if _can_parse_latex and '\\' in final:
                expr = parse_latex(final)
            else:
                expr = parse_expr(prepare_expr(final), transformations=_transformations)
        else:
            if data.a is not None:
                a, H = data.a, 0.0
            elif data.altura is not None:
                H = data.altura
                a = -4 * H / (W**2)
            else:
                raise ValueError("Envía 'equation' o bien 'a' o 'altura'")
            expr = a*x**2 + H

        # ── 2) Simplificar y generar LaTeX
        expr_s  = sp.simplify(expr)
        eq_text = r"y(x) = " + sp.latex(expr_s)

        # ── 3) Hallar vértice y foco (parábola vertical)
        poly = sp.Poly(expr_s, x)
        a_coef, b_coef, _ = poly.all_coeffs()  # [a, b, c]
        xv = -b_coef / (2*a_coef)
        yv = expr_s.subs(x, xv)
        p  = 1/(4*a_coef)           # distancia foco-vértice
        xf, yf = xv, yv + p

        # ── 4) Área y lista de puntos
        area = float(sp.integrate(expr_s, (x, -W/2, W/2)))
        xs = np.linspace(-W/2, W/2, 50)
        pts = [(float(xi), float(expr_s.subs(x, xi)), 0.0) for xi in xs]

        return {
            "equation": eq_text,
            "area_total": round(area, 2),
            "points": pts,
            "vertex": (float(xv), float(yv), 0.0),
            "focus":  (float(xf), float(yf), 0.0),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
