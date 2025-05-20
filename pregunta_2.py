import numpy as np
import matplotlib.pyplot as plt

# Método de Thomas para resolver matrices tridiagonales
def metodo_thomas(a, b, c, d):
    n = len(d)
    c_mod = np.zeros(n-1)
    d_mod = np.zeros(n)

    # Primer paso de eliminación hacia adelante
    c_mod[0] = c[0] / b[0]
    d_mod[0] = d[0] / b[0]

    for i in range(1, n-1):
        divisor = b[i] - a[i-1] * c_mod[i-1]
        c_mod[i] = c[i] / divisor
        d_mod[i] = (d[i] - a[i-1] * d_mod[i-1]) / divisor

    # Último valor de d
    d_mod[-1] = (d[-1] - a[-2] * d_mod[-2]) / (b[-1] - a[-2] * c_mod[-2])

    # Sustitución hacia atrás
    x = np.zeros(n)
    x[-1] = d_mod[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_mod[i] - c_mod[i] * x[i+1]

    return x

# Función que resuelve el problema usando diferencias finitas
def edo2(p_func, q_func, r_func, h, x_ini, x_fin, y_ini, y_fin):
    n = int((x_fin - x_ini) / h)  # número de puntos
    puntos_x = np.linspace(x_ini, x_fin, n+1)  # nodos del intervalo
    nodos_internos = puntos_x[1:-1]  # sin incluir los extremos

    p_vals = p_func(nodos_internos)
    q_vals = q_func(nodos_internos)
    r_vals = r_func(nodos_internos)

    # Coeficientes del sistema tridiagonal
    a = -0.5 * h * p_vals - 1
    b = 2 + h**2 * q_vals
    c = 0.5 * h * p_vals - 1

    lado_derecho = -h**2 * r_vals

    # Ajustamos el primer y último elemento con las condiciones de frontera
    lado_derecho[0] -= a[0] * y_ini
    lado_derecho[-1] -= c[-1] * y_fin

    # Aplicamos método de Thomas
    subdiag = a[1:]
    diag = b
    superdiag = c[:-1]
    solucion_interna = metodo_thomas(subdiag, diag, superdiag, lado_derecho)

    # Agregamos las condiciones de frontera a la solución
    solucion_total = np.concatenate(([y_ini], solucion_interna, [y_fin]))

    return puntos_x, solucion_total

# Funciones dadas en el problema
def funcion_p(x): return -1 / x
def funcion_q(x): return (1 / (4 * x**2)) - 1
def funcion_r(x): return np.zeros_like(x)

# Solución exacta para comparación
def solucion_exacta(x): return np.sin(6 - x) / (np.sin(5) * np.sqrt(x))

# Función principal para graficar
def graficar_soluciones():
    pasos = [1, 0.5, 0.2, 0.1, 0.01]
    x_ini, x_fin = 1, 6
    y_ini, y_fin = 1, 0

    # Crear vector denso para la curva exacta
    x_denso = np.linspace(x_ini, x_fin, 500)
    plt.figure(figsize=(10, 6))
    plt.plot(x_denso, solucion_exacta(x_denso), label='Solución exacta', linewidth=2, color='black')

    # Calcular y graficar soluciones numéricas con diferentes h
    for h in pasos:
        x_vals, y_vals = edo2(funcion_p, funcion_q, funcion_r, h, x_ini, x_fin, y_ini, y_fin)
        plt.plot(x_vals, y_vals, label=f'Aproximación h = {h}')

    # Personalizar gráfico
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Aproximación usando diferencias finitas')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

graficar_soluciones()
