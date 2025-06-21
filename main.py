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
from sympy.parsing.latex import parse_latex
from sympy import Eq, symbols, solve, Matrix, sqrt

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

def identify_conic(expr, x, y):
    expr = sp.expand(expr)
    
    A = expr.coeff(x**2)
    B = expr.coeff(x*y)
    C = expr.coeff(y**2)
    D = expr.coeff(x)
    E = expr.coeff(y)
    F = expr.subs({x: 0, y: 0})

    def to_float(val):
        try:
            return float(val.evalf()) if val != 0 else 0.0
        except (TypeError, AttributeError):
            return 0.0
    A, B, C, D, E, F = map(to_float, [A, B, C, D, E, F])

    discriminant = B**2 - 4*A*C

    params = {}
    tipo = "unknown"
    centro = (0.0, 0.0)

    if B != 0:
        theta = sp.atan2(B, A - C) / 2
        x_rot = x * sp.cos(theta) + y * sp.sin(theta)
        y_rot = -x * sp.sin(theta) + y * sp.cos(theta)
        expr_rot = expr.subs({x: x_rot, y: y_rot}).expand()
        A = expr_rot.coeff(x**2)
        C = expr_rot.coeff(y**2)
        D = expr_rot.coeff(x)
        E = expr_rot.coeff(y)
        F = expr_rot.subs({x: 0, y: 0})
        A, C, D, E, F = map(to_float, [A, C, D, E, F])
        discriminant = -4*A*C
        params['rotacion'] = float(theta)

    if abs(A) < 1e-10 and abs(C) < 1e-10:
        if abs(E) > 1e-10 or abs(D) > 1e-10:
            tipo = "parabola"
            if abs(E + 1) < 1e-10:
                a_parabola = A
                b_parabola = D
                if abs(a_parabola) > 1e-10:
                    p = abs(1/(4*a_parabola))
                    cx = -b_parabola/(2*a_parabola)
                    cy = -p
                    centro = (cx, cy)
                    params = {
                        "foco_distancia": p,
                        "directriz": f"y = {cy + p}",
                        "vertice": (cx, cy)
                    }
                else:
                    params = {"foco_distancia": 0.0, "directriz": "indefinida"}
            else:
                params = {"foco_distancia": 0.0, "directriz": "indefinida"}
    elif discriminant < -1e-10:
        if abs(A - C) < 1e-10:
            tipo = "circle"
            cx = -D/(2*A) if abs(A) > 1e-10 else 0.0
            cy = -E/(2*A) if abs(A) > 1e-10 else 0.0
            centro = (cx, cy)
            try:
                radius = sqrt((D**2 + E**2 - 4*A*F)/(4*A**2))
                params = {"centro": centro, "radio": float(radius)}
            except (ValueError, TypeError):
                raise ValueError("Ecuación no representa un círculo válido")
        else:
            tipo = "ellipse"
            cx = -D/(2*A) if abs(A) > 1e-10 else 0.0
            cy = -E/(2*C) if abs(C) > 1e-10 else 0.0
            centro = (cx, cy)
            try:
                a_axis = sqrt(-F/A) if A != 0 and F/A < 0 else 1.0
                b_axis = sqrt(-F/C) if C != 0 and F/C < 0 else 1.0
                params = {"centro": centro, "ejes": [float(a_axis), float(b_axis)]}
            except (ValueError, TypeError):
                raise ValueError("Ecuación no representa una elipse válida")
    elif abs(discriminant) < 1e-10:
        tipo = "parabola"
        cx = -D/(2*A) if abs(A) > 1e-10 else 0.0
        cy = -E/(2*C) if abs(C) > 1e-10 else 0.0
        centro = (cx, cy)
        try:
            p = abs(1/(4*A)) if abs(A) > 1e-10 else abs(1/(4*C))
            params = {"foco_distancia": float(p), "directriz": f"y = {cy + p}"}
        except (ValueError, TypeError):
            params = {"foco_distancia": 0.0, "directriz": "indefinida"}
    elif discriminant > 1e-10:
        tipo = "hyperbola"
        cx = -D/(2*A) if abs(A) > 1e-10 else 0.0
        cy = -E/(2*C) if abs(C) > 1e-10 else 0.0
        centro = (cx, cy)
        try:
            a_axis = sqrt(F/A) if A != 0 and F/A > 0 else 1.0
            b_axis = sqrt(-F/C) if C != 0 and F/C < 0 else 1.0
            params = {"centro": centro, "ejes": [float(a_axis), float(b_axis)]}
        except (ValueError, TypeError):
            raise ValueError("Ecuación no representa una hipérbola válida")
    else:
        raise ValueError("Ecuación no representa una sección cónica válida")

    return tipo, centro, params

@app.post("/calcular", response_model=ResponseData)
async def calcular(data: RequestData):
    try:
        x, y = symbols('x y')

        if data.equation:
            if '=' not in data.equation:
                raise ValueError("La ecuación debe contener '='")
            try:
                expr = parse_latex(data.equation)
                lhs, rhs = data.equation.split('=', 1)
                expr_to_analyze = parse_expr(prepare(lhs) + " - (" + prepare(rhs) + ")", transformations=_transformations)
            except Exception:
                lhs, rhs = data.equation.split('=', 1)
                expr_to_analyze = parse_expr(prepare(lhs) + " - (" + prepare(rhs) + ")", transformations=_transformations)
        else:
            if data.a is not None:
                if abs(data.a) < 1e-10:
                    raise ValueError("El coeficiente 'a' no puede ser cero")
                a = data.a
            elif data.altura is not None and data.ancho is not None:
                if abs(data.ancho) < 1e-10:
                    raise ValueError("El ancho no puede ser cero")
                if data.altura == 0:
                    raise ValueError("La altura no puede ser cero")
                a = -4 * data.altura / (data.ancho ** 2)
            else:
                raise ValueError("Debe enviar 'equation' o 'a' o (altura y ancho)")
            expr_to_analyze = a * x**2 - y

        tipo, centro, params = identify_conic(expr_to_analyze, x, y)

        # Generar puntos, manejar soluciones complejas tomando la parte real
        xs = np.linspace(centro[0] - 5, centro[0] + 5, 200)
        ys = []
        for xi in xs:
            sol = solve(expr_to_analyze.subs(x, xi), y, domain=sp.S.Reals)  # Restringir a reales
            if sol:
                yi = float(sol[0].evalf()) if sol[0].is_real else 0.0
                ys.append(yi)
            else:
                ys.append(0.0)

        pts = [(float(xi), float(yi), 0.0) for xi, yi in zip(xs, ys) if yi is not None]

        expr_eq = Eq(expr_to_analyze, 0)
        expr_latex = sp.latex(expr_eq)

        return ResponseData(
            equation="\\text{Ecuación: } " + expr_latex,
            tipo=tipo,
            parametros=params,
            points=pts
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)