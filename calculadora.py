import sympy as sp
import numpy as np

def expr_parabola_canonica(H: float, W: float):
    """
    Retorna la expresión simplificada de la parábola en forma canónica
    y su representación en LaTeX: y(x) = a x^2 + H.
    """
    x = sp.symbols('x')
    a = -4 * H / (W**2)
    expr = a * x**2 + H
    expr_s = sp.simplify(expr)
    latex_body = sp.latex(expr_s)
    text = r"y(x) = " + latex_body
    return expr_s, text

def genera_puntos(expr, W: float, num: int = 50):
    """
    Genera una lista de puntos (x, y, 0.0) para graficar la curva
    desde -W/2 hasta W/2.
    """
    x = sp.symbols('x')
    xs = np.linspace(-W/2, W/2, num)
    pts = [(float(xi), float(expr.subs(x, xi)), 0.0) for xi in xs]
    return pts