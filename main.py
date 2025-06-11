from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from calculadora import calcular_area_muro
import sympy as sp
import numpy as np

app = FastAPI()

class RequestData(BaseModel):
    # Modo medidas
    altura: float = None
    ancho: float = None
    a: float = None
    # Modo ecuación
    equation: str = None  # por ejemplo "0.5*x**2 + 2"

class ResponseData(BaseModel):
    area_total: float
    # lista de puntos para graficar curve en 3D (x,y,z)
    points: list[tuple[float,float,float]]

@app.post("/calcular", response_model=ResponseData)
def calcular(data: RequestData):
    try:
        # Si viene equation, parseamos
        if data.equation:
            if data.ancho is None:
                raise ValueError("Debes darme un 'ancho' para integrar tu ecuación")
            # Defino x simbólico y expr
            x = sp.symbols('x')
            expr = sp.sympify(data.equation)
            # Integro de -ancho/2 a ancho/2
            area = float(sp.integrate(expr, (x, -data.ancho/2, data.ancho/2)))
            # Genero puntos para graficar
            xs = np.linspace(-data.ancho/2, data.ancho/2, 50)
            ys = [float(expr.subs(x, xi)) for xi in xs]
            # Para 3D, z será la "altura" o la coordenada perpendicular: 
            # aquí simplemente pongo z = 0 (superficie curva en x–y)
            pts = [(float(xi), float(yi), 0.0) for xi, yi in zip(xs, ys)]
        else:
            # modo clásico: medidas + coeficiente a
            if data.altura is None or data.ancho is None:
                raise ValueError("Altura y ancho son obligatorios")
            area = calcular_area_muro(data.altura, data.ancho, data.a)
            # En modo medidas devolvemos la parábola y= a x²
            if data.a is not None:
                xs = np.linspace(-data.ancho/2, data.ancho/2, 50)
                ys = [data.a * (xi**2) for xi in xs]
            else:
                xs = [0.0]
                ys = [0.0]
            pts = [(float(xi), float(yi), float(data.altura)) for xi, yi in zip(xs, ys)]
            # (aquí z lo pongo como la altura total del muro, para ver un perfil en 3D)
        return {"area_total": round(area,2), "points": pts}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
