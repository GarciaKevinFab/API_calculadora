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

# Combinamos los transformadores
_transformations = standard_transformations + (implicit_multiplication_application,)

app = FastAPI()

class RequestData(BaseModel):
    equation: str   | None = None  # LaTeX o texto
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
        W = data.ancho
        x, y = sp.symbols('x y')

        if data.equation:
            raw = data.equation.strip()

            # Caso implicita con '='
            if '=' in raw:
                left, right = raw.split('=', 1)

                # Parseamos cada lado según el caso
                if _can_parse_latex and '\\' in raw:
                    L = parse_latex(left)
                    R = parse_latex(right)
                else:
                    # reemplazamos '^' por '**' y permitimos '6x' => '6*x'
                    L = parse_expr(left.replace('^','**'), transformations=_transformations)
                    R = parse_expr(right.replace('^','**'), transformations=_transformations)

                # Ecuación L - R = 0
                eq0 = L - R
                sols = sp.solve(eq0, y)
                if not sols:
                    raise ValueError("No pude despejar 'y' de la ecuación")
                expr = sols[0]
                eq_text = f"y(x) = {sp.sstr(expr)}"

            else:
                # Función explícita sin '='
                if _can_parse_latex and '\\' in raw:
                    expr = parse_latex(raw)
                else:
                    expr = parse_expr(raw.replace('^','**'), transformations=_transformations)
                eq_text = f"y(x) = {raw}"

        else:
            # Modo parámetro
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

        # Área e puntos
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
