import numpy as np
import matplotlib.pyplot as plt

# Funciones dadas en el problema
def funcion_p(x): return -1 / x
def funcion_q(x): return (1 / (4 * x**2)) - 1
def funcion_r(x): return np.zeros_like(x)

# Solución exacta para comparar
def solucion_exacta(x): return np.sin(6 - x) / (np.sin(5) * np.sqrt(x))

# Método de Thomas para sistemas tridiagonales
def metodo_thomas(a, b, c, d):
    n = len(d)
    c_mod = np.zeros(n-1)
    d_mod = np.zeros(n)
    c_mod[0] = c[0] / b[0]
    d_mod[0] = d[0] / b[0]
    for i in range(1, n-1):
        divisor = b[i] - a[i-1] * c_mod[i-1]
        c_mod[i] = c[i] / divisor
        d_mod[i] = (d[i] - a[i-1] * d_mod[i-1]) / divisor
    d_mod[-1] = (d[-1] - a[-2] * d_mod[-2]) / (b[-1] - a[-2] * c_mod[-2])
    x = np.zeros(n)
    x[-1] = d_mod[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_mod[i] - c_mod[i] * x[i+1]
    return x

# Resolver la EDO usando diferencias finitas
def edo2(p_func, q_func, r_func, h, x_ini, x_fin, y_ini, y_fin):
    n = int((x_fin - x_ini) / h)
    puntos_x = np.linspace(x_ini, x_fin, n+1)
    nodos_internos = puntos_x[1:-1]
    p_vals = p_func(nodos_internos)
    q_vals = q_func(nodos_internos)
    r_vals = r_func(nodos_internos)
    a = -0.5 * h * p_vals - 1
    b = 2 + h**2 * q_vals
    c = 0.5 * h * p_vals - 1
    lado_derecho = -h**2 * r_vals
    lado_derecho[0] -= a[0] * y_ini
    lado_derecho[-1] -= c[-1] * y_fin
    subdiag = a[1:]
    diag = b
    superdiag = c[:-1]
    solucion_interna = metodo_thomas(subdiag, diag, superdiag, lado_derecho)
    return puntos_x, np.concatenate(([y_ini], solucion_interna, [y_fin]))

# Construir los trazadores cúbicos naturales
def trazador_cubico_natural(x, y):
    n = len(x) - 1
    h = np.diff(x)
    a = h[:-1]
    b = 2 * (h[:-1] + h[1:])
    c = h[1:]
    d = 6 * ((y[2:] - y[1:-1]) / h[1:] - (y[1:-1] - y[:-2]) / h[:-1])
    M = np.zeros(n + 1)
    M[1:n] = metodo_thomas(a, b, c, d)
    coeficientes = []
    for i in range(n):
        hi = h[i]
        ai = (M[i+1] - M[i]) / (6 * hi)
        bi = M[i] / 2
        ci = (y[i+1] - y[i]) / hi - (2 * hi * M[i] + hi * M[i+1]) / 6
        di = y[i]
        coeficientes.append((ai, bi, ci, di))
    return coeficientes

# Evaluar cada polinomio cúbico en su subintervalo
def evaluar_trazador(xi, coefs, x_eval):
    ai, bi, ci, di = coefs
    return ai * (x_eval - xi)**3 + bi * (x_eval - xi)**2 + ci * (x_eval - xi) + di

# Gráfica final
def graficar_trazador_vs_original():
    h = 1
    x_ini, x_fin = 1, 6
    y_ini, y_fin = 1, 0

    # Obtener nodos y valores de la solución numérica
    x_vals, y_vals = edo2(funcion_p, funcion_q, funcion_r, h, x_ini, x_fin, y_ini, y_fin)

    # Obtener coeficientes del trazador cúbico natural
    coef_tramos = trazador_cubico_natural(x_vals, y_vals)

    # Preparar evaluación densa
    x_denso = np.linspace(x_ini, x_fin, 500)
    y_trazador = np.zeros_like(x_denso)

    for i in range(len(coef_tramos)):
        idx = np.where((x_denso >= x_vals[i]) & (x_denso <= x_vals[i+1]))
        y_trazador[idx] = evaluar_trazador(x_vals[i], coef_tramos[i], x_denso[idx])

    # Solución exacta para comparar
    y_exacta = solucion_exacta(x_denso)

    # Graficar resultados
    plt.figure(figsize=(10, 6))
    plt.plot(x_denso, y_exacta, label='Solución original', color='black', linewidth=2)
    plt.plot(x_denso, y_trazador, label='Trazador cúbico natural', linestyle='--', color='blue')
    plt.scatter(x_vals, y_vals, color='red', label='Nodos de interpolación')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Trazador Cúbico Natural vs Solución Original')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

graficar_trazador_vs_original()
