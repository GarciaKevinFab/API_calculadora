import sympy as sp
import numpy as np

def expr_parabola_canonica(H: float, W: float):
    x = sp.symbols('x')
    a = -4 * H / (W**2)
    expr = a * x**2 + H
    text = f"y(x) = {sp.simplify(expr)}"
    return expr, text

def genera_puntos(expr, W: float, num: int = 50):
    x = sp.symbols('x')
    xs = np.linspace(-W/2, W/2, num)
    pts = [(float(xi), float(expr.subs(x, xi)), 0.0) for xi in xs]
    return pts
