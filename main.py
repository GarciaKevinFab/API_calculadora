import re
from fastapi import FastAPI, HTTPException
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

class RequestData(BaseModel):
    equation: str | None = None
    a: float | None = None
    altura: float | None = None
    ancho: float

class ResponseData(BaseModel):
    equation: str
    area_total: float
    points: list[tuple[float, float, float]]

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

        # 1) Obtener expr
        if data.equation:
            raw = data.equation.strip()
            if '=' in raw:
                Ls, Rs = raw.split('=',1)
                if _can_parse_latex and '\\' in raw:
                    L = parse_latex(Ls); R = parse_latex(Rs)
                else:
                    L = parse_expr(prepare_expr(Ls), transformations=_transformations)
                    R = parse_expr(prepare_expr(Rs), transformations=_transformations)
                eq0 = L - R
                if isinstance(eq0, sp.Derivative):
                    sol = sp.dsolve(eq0, y); expr = sol.rhs
                else:
                    sols = sp.solve(eq0, y)
                    if not sols: raise ValueError("No pude despejar 'y'")
                    expr = sols[0]
            else:
                expr = (_can_parse_latex and '\\' in raw
                        and parse_latex(raw)
                        or parse_expr(prepare_expr(raw), transformations=_transformations))
        else:
            if data.a is not None:
                a, H = data.a, 0.0
            elif data.altura is not None:
                H = data.altura
                a = -4 * H / (W**2)
            else:
                raise ValueError("Envía 'equation' o 'a' o 'altura'")
            expr = a*x**2 + H

        # 2) Simplificar y LaTeX
        expr_s = sp.simplify(expr)
        eq_text = r"y(x) = " + sp.latex(expr_s)

        # 3) Área y puntos
        area = float(sp.integrate(expr_s, (x, -W/2, W/2)))
        xs = np.linspace(-W/2, W/2, 50)
        pts = [(float(xi), float(expr_s.subs(x, xi)), 0.0) for xi in xs]

        return {"equation": eq_text, "area_total": round(area,2), "points": pts}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))