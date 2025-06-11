import sympy as sp
import numpy as np

def expr_parabola_canonica(H: float, W: float):
    """
    Construye la parábola canónica de altura H y ancho W:
      y(x) = -4H/W^2 * x^2 + H
    Devuelve (expr_simpy, texto_para_cliente).
    """
    x = sp.symbols('x')
    a = -4 * H / (W**2)
    expr = a * x**2 + H
    text = f"y(x) = {sp.simplify(expr)}"
    return expr, text

def integra_area(expr, W: float):
    """
    Integra la expr de -W/2 a +W/2 y devuelve el área (float).
    """
    x = sp.symbols('x')
    area = float(sp.integrate(expr, (x, -W/2, W/2)))
    return area

def genera_puntos(expr, W: float, num: int = 50):
    """
    Genera una lista de puntos (x, y(x), 0) equiespaciados en [-W/2, W/2].
    num = número de muestras.
    """
    x = sp.symbols('x')
    xs = np.linspace(-W/2, W/2, num)
    ys = [float(expr.subs(x, xi)) for xi in xs]
    return [(float(xi), float(yi), 0.0) for xi, yi in zip(xs, ys)]