import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import sympy as sp
from typing import Optional
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)
from sympy.geometry import Conic
from sympy import Eq

_transformations = standard_transformations + (implicit_multiplication_application,)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RequestData(BaseModel):
    equation: Optional[str] = None
    a: Optional[float] = None
    altura: Optional[float] = None
    ancho: Optional[float] = None

class ResponseData(BaseModel):
    equation: str
    tipo: str
    parametros: dict
    points: list[tuple[float, float, float]]

def prepare(s: str) -> str:
    t = s.replace('^', '**')
    t = re.sub(r'(?<=\d)(?=[A-Za-z])', '*', t)
    t = re.sub(r'(?<=[A-Za-z])(?=\d)', '*', t)
    t = t.replace(' ', '')
    return t

@app.post("/calcular", response_model=ResponseData)
def calcular(data: RequestData):
    try:
        x, y = sp.symbols('x y')

        # Modo ecuación
        if data.equation:
            if '=' not in data.equation:
                raise ValueError("La ecuación debe contener '='.")
            lhs, rhs = data.equation.split('=', 1)
            expr = parse_expr(prepare(lhs) + "-(" + prepare(rhs) + ")", transformations=_transformations)
        # Modo parámetro (a o altura + ancho)
        else:
            if data.a is not None:
                a = data.a
            elif data.altura is not None and data.ancho is not None:
                a = -4 * data.altura / (data.ancho ** 2)
            else:
                raise ValueError("Debe enviar una 'equation' o bien 'a' o (altura y ancho)")
            expr = a * x**2 - y

        con = Conic(Eq(expr, 0))
        tipo = con.conic_type
        centro = tuple(map(float, con.center)) if hasattr(con, 'center') else (0.0, 0.0)
        params = {}

        if tipo == 'circle':
            r = float(con.radius)
            params = {"centro": centro, "radio": r}
        elif tipo == 'parabola':
            d = float(con.parameter)
            params = {"foco_distancia": d, "directriz": str(con.directrix)}
        elif tipo == 'ellipse':
            a, b = float(con.a), float(con.b)
            params = {"centro": centro, "ejes": [a, b]}
        elif tipo == 'hyperbola':
            a, b = float(con.a), float(con.b)
            params = {"centro": centro, "ejes": [a, b]}

        # Generar puntos Y de verdad (solución de y)
        xs = np.linspace(centro[0] - 5, centro[0] + 5, 200)
        ys = []
        for xi in xs:
            sol = sp.solve(expr.subs(x, xi), y)
            if sol:
                yi = float(sol[0].evalf())
                ys.append(yi)
            else:
                ys.append(0.0)

        pts = [(float(xi), float(yi), 0.0) for xi, yi in zip(xs, ys)]

        # Generar expresión para mostrar
        expr_eq = Eq(expr, 0)
        expr_latex = sp.latex(expr_eq)

        return ResponseData(
            equation="\\text{Ecuación: } " + expr_latex,
            tipo=tipo,
            parametros=params,
            points=pts
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
