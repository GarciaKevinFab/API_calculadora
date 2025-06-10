# Lógica pura de cálculo

def calcular_area_muro(altura: float, ancho: float, a: float = None) -> float:
    area_rect = altura * ancho
    if a is not None:
        # Integral de a*x^2 de -ancho/2 a ancho/2
        area_parab = (a * (ancho/2)**3 / 3) * 2
        return area_rect - abs(area_parab)
    return area_rect