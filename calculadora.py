import sympy as sp
import numpy as np

def expr_parabola_canonica(H: float, W: float):
    x = sp.symbols('x')
    a = -4 * H / (W**2)
    expr = a * x**2 + H
    text = f"y(x) = {sp.simplify(expr)}"
    return expr, text

def integra_area(expr, W: float):
    x = sp.symbols('x')
    return float(sp.integrate(expr, (x, -W/2, W/2)))

def genera_puntos(expr, W: float, num: int = 50):
    x = sp.symbols('x')
    xs = np.linspace(-W/2, W/2, num)
    ys = [float(expr.subs(x, xi)) for xi in xs]
    return [(float(xi), float(yi), 0.0) for xi, yi in zip(xs, ys)]
