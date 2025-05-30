import numpy as np
import matplotlib.pyplot as plt
import sympy as sp

# Resuelve un sistema de ecuaciones lineales con matriz tridiagonal
def metodo_thomas(a, b, c, d):
    n = len(d)
    c_mod = np.zeros(n-1)  # vector para la diagonal superior modificada
    d_mod = np.zeros(n)    # vector para el lado derecho modificado

    # Primer paso: modificar el primer valor
    c_mod[0] = c[0] / b[0]
    d_mod[0] = d[0] / b[0]

    # Eliminación hacia adelante
    for i in range(1, n-1):
        divisor = b[i] - a[i-1] * c_mod[i-1]
        c_mod[i] = c[i] / divisor
        d_mod[i] = (d[i] - a[i-1] * d_mod[i-1]) / divisor

    # Última fila
    d_mod[-1] = (d[-1] - a[-2] * d_mod[-2]) / (b[-1] - a[-2] * c_mod[-2])

    # Sustitución hacia atrás
    x = np.zeros(n)
    x[-1] = d_mod[-1]
    for i in range(n-2, -1, -1):
        x[i] = d_mod[i] - c_mod[i] * x[i+1]

    return x

# Resuelve una EDO usando el método de diferencias finitas
def edo2(p_func, q_func, r_func, h, x_ini, x_fin, y_ini, y_fin):
    n = int((x_fin - x_ini) / h)
    puntos_x = np.linspace(x_ini, x_fin, n+1)     # nodos del intervalo
    nodos_internos = puntos_x[1:-1]               # quitamos los extremos

    # Evaluamos las funciones en los puntos internos
    p_vals = p_func(nodos_internos)
    q_vals = q_func(nodos_internos)
    r_vals = r_func(nodos_internos)

    # Construcción de las diagonales
    a = -0.5 * h * p_vals - 1
    b = 2 + h**2 * q_vals
    c = 0.5 * h * p_vals - 1

    # Lado derecho del sistema
    lado_derecho = -h**2 * r_vals
    lado_derecho[0] -= a[0] * y_ini       # condición inicial
    lado_derecho[-1] -= c[-1] * y_fin     # condición final

    # Aplicamos el método de Thomas
    subdiag = a[1:]
    diag = b
    superdiag = c[:-1]
    solucion_interna = metodo_thomas(subdiag, diag, superdiag, lado_derecho)

    # Agregamos los extremos conocidos (condiciones de frontera)
    solucion_total = np.concatenate(([y_ini], solucion_interna, [y_fin]))
    return puntos_x, solucion_total

def funcion_p(x): return -1 / x
def funcion_q(x): return (1 / (4 * x**2)) - 1
def funcion_r(x): return np.zeros_like(x)
def solucion_exacta(x): return np.sin(6 - x) / (np.sin(5) * np.sqrt(x))

# Construye el polinomio de Lagrange a partir de una lista de puntos
def polinomio_lagrange(puntos):
    x = sp.Symbol('x')  # Variable simbólica
    n = len(puntos)
    polinomio = 0  # Inicializamos el polinomio completo

    # Recorremos cada punto para construir su término L_k(x)
    for i in range(n):
        xi, yi = puntos[i]
        L = 1  # Base L_k(x)

        for j in range(n):
            if j != i:
                xj, _ = puntos[j]
                L *= (x - xj) / (xi - xj)

        polinomio += yi * L

    # Simplificamos la expresión simbólica del polinomio
    return sp.simplify(polinomio)

# Convierte el polinomio simbólico a función de numpy para graficar
def evaluar_polinomio_lagrange(expr, valores_x):
    x = sp.Symbol('x')
    f_lambdified = sp.lambdify(x, expr, modules=['numpy'])
    return f_lambdified(valores_x)

def graficar_comparacion():
    h = 1
    x_ini, x_fin = 1, 6
    y_ini, y_fin = 1, 0

    # Obtenemos la solución numérica
    puntos_x, solucion_y = edo2(funcion_p, funcion_q, funcion_r, h, x_ini, x_fin, y_ini, y_fin)
    puntos_interpolacion = list(zip(puntos_x, solucion_y))

    # Construimos el polinomio de Lagrange
    pol_lagrange = polinomio_lagrange(puntos_interpolacion)

    # Evaluamos ambas funciones en un conjunto denso de x
    x_vals_denso = np.linspace(x_ini, x_fin, 500)
    y_exacta = solucion_exacta(x_vals_denso)
    y_interp = evaluar_polinomio_lagrange(pol_lagrange, x_vals_denso)

    # Gráfica
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals_denso, y_exacta, label='Solución original (Problema 2)', linewidth=2, color='black')
    plt.plot(x_vals_denso, y_interp, label='Polinomio de Lagrange', linestyle='--', linewidth=2, color='blue')
    plt.scatter(*zip(*puntos_interpolacion), color='red', zorder=5, label='Puntos de interpolación')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Comparación: Solución original vs. Interpolación de Lagrange')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Mostramos el polinomio en consola
    print("\nPolinomio de Lagrange simplificado:")
    sp.pprint(pol_lagrange, use_unicode=True)

# Ejecutar función principal
graficar_comparacion()
