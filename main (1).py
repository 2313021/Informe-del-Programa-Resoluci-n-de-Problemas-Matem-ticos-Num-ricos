#INTEGRANTES 
#JOHAN SEBASTIAN LAVERDE PINEDA 2266278-3743
#JESUS ESTENLLOS LOAIZA SERRANO 2313021

import numpy as np  # Importa la biblioteca numpy para operaciones numéricas y manejo de arrays
import matplotlib.pyplot as plt  # Importa matplotlib para crear gráficos, usando el alias plt
from math import factorial  # Importa la función factorial del módulo math para calcular factoriales
from scipy.optimize import fsolve  # Importa fsolve de scipy.optimize para resolver ecuaciones no lineales
from scipy.special import factorial
from scipy.misc import derivative

def menu():
    print("Seleccione una opción:")
    print("1. Inicio")
    print("2. Tylor")
    print("3. El método de Newton")
    print("4. Método de diferencias finitas")
    print("5. Sistemas de ecuaciones no lineales")
    print("6. Sistemas de ecuaciones lineales")
    print("7. ED Primer orden")
    print("8. ED Transformada de LAPLACE")
def main():
    while True:
        menu()
        opcion = input("Ingrese una opción: ")

        if opcion == '2':
            print("\n")
            print("METODO DE TAYLOR")
            print("\n")

            def tipo_fun():
                print("1. sin")
                print("2. cos")
                print("3. exp")
                
                while True:
                    opc = input("Seleccione la funcion: ")
                    if opc == '1':
                        return np.sin
                    elif opc == '2':
                        return np.cos
                    elif opc == '3':
                        return np.exp
                    else:
                        print("Opción inválida. Intenta nuevamente.")

            # Serie de Taylor mejorada
            def taylor_series(func, x0, terms, x_vals):
                series = np.zeros_like(x_vals)
                for n in range(terms):
                    deriv = derivative(func, x0, n=n, order=2*n+1)
                    term = (deriv / factorial(n)) * (x_vals - x0) ** n
                    series += term
                return series

            # Parámetros ingresados por el usuario
            print("Ingrese los siguientes datos:")
            func = tipo_fun()

            x0 = float(input("Punto de expansión (x0): "))
            terms = int(input("Número de términos de Taylor: "))
            x_min = float(input("Límite inferior del rango de x: "))
            x_max = float(input("Límite superior del rango de x: "))

            # Rango de valores de x
            x_vals = np.linspace(x_min, x_max, 500)

            # Cálculo de la serie de Taylor
            taylor_approx = taylor_series(func, x0, terms, x_vals)

            # Gráfica mejorada
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, func(x_vals), label="Función original", color="blue")
            plt.plot(x_vals, taylor_approx, label=f"Taylor (n={terms})", linestyle="--", color="red")
            plt.axvline(x=x0, color="gray", linestyle=":", label=f"Punto de expansión x0={x0}")
            plt.title("Aproximación con la Serie de Taylor")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            plt.grid(True, linestyle='--', linewidth=0.5)
            plt.show()

        if opcion == '3':
            print("\n")
            print("METODO DE NEWTON")
            print("\n")
            # Método de Newton para encontrar las raíces
            def newton_method(func, dfunc, x0, tol=1e-6, max_iter=100):
                """
                Aplica el método de Newton para encontrar la raíz de una función.
                - func: función a la que se le quiere encontrar la raíz.
                - dfunc: derivada de la función.
                - x0: punto inicial.
                - tol: tolerancia para la convergencia (por defecto 1e-6).
                - max_iter: número máximo de iteraciones (por defecto 100).
                """
                x_vals = [x0]
                for _ in range(max_iter):
                    fx = func(x0)
                    dfx = dfunc(x0)
                    
                    # Prevención de división por cero
                    if dfx == 0:
                        print("La derivada es cero, el método no puede continuar.")
                        break
                    
                    # Método de Newton
                    x_new = x0 - fx / dfx
                    x_vals.append(x_new)
                    
                    # Verificar la convergencia
                    if abs(x_new - x0) < tol:
                        break
                    x0 = x_new
                
                return x_vals

            # Definición de la función y su derivada
            def func(x):
                return np.sin(x)  # Puedes cambiar esta función a np.cos(x), np.exp(x), etc.

            def dfunc(x):
                return np.cos(x)  # Derivada de np.sin(x), si cambias la función, cambia también su derivada.

            # Parámetros ingresados por el usuario
            x0 = float(input("Ingresa el punto inicial (x0): "))
            max_iter = int(input("Número máximo de iteraciones: "))
            x_min = float(input("Ingresa el límite inferior del rango de x: "))
            x_max = float(input("Ingresa el límite superior del rango de x: "))

            # Ejecutar el método de Newton
            x_vals = newton_method(func, dfunc, x0, max_iter=max_iter)

            # Graficar la función y la secuencia de aproximaciones
            x_range = np.linspace(x_min, x_max, 500)
            y_vals = func(x_range)

            plt.figure(figsize=(8, 5))
            plt.plot(x_range, y_vals, label="Función: f(x)", color="blue")
            plt.scatter(x_vals, func(np.array(x_vals)), color="red", label="Aproximaciones (Newton)", zorder=5)
            plt.axhline(0, color="black",linewidth=0.5)
            plt.axvline(0, color="black",linewidth=0.5)
            plt.title("Método de Newton")
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.legend()
            plt.grid(True)
            plt.show()

            # Mostrar la secuencia de aproximaciones
            print(f"Raíz aproximada encontrada: {x_vals[-1]}")
            print(f"Secuencia de aproximaciones: {x_vals}")    

        if opcion == '4':
            print("\n")
            print("METODO DE DIFERENCIAS FINITAS")
            print("\n")
            # Solicitar datos al usuario
            a = float(input("Ingrese el inicio del intervalo (a): "))
            b = float(input("Ingrese el final del intervalo (b): "))
            n = int(input("Ingrese el número de puntos (n): "))

            # Definir la función
            def f(x):
                return np.sin(x)  # Puedes cambiarla por cualquier otra función

            # Puntos en el intervalo
            x = np.linspace(a, b, n)
            h = x[1] - x[0]  # Paso 

            # Aproximación de la derivada usando diferencias finitas centradas
            f_prime = np.zeros_like(x)
            f_prime[1:-1] = (f(x[2:]) - f(x[:-2])) / (2 * h)

            # Derivada exacta (para comparación)
            def f_exact_prime(x):
                return np.cos(x)  # Derivada de sin(x)

            f_exact = f_exact_prime(x)

            # Gráfica
            plt.figure(figsize=(8, 6))
            plt.plot(x, f(x), label="f(x) = sin(x)", color="blue")
            plt.plot(x, f_prime, label="Diferencias finitas", color="red", linestyle="--")
            plt.plot(x, f_exact, label="Derivada exacta", color="green", linestyle=":")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Aproximación de la derivada con diferencias finitas")
            plt.grid()
            plt.show()

        if opcion == '5':
            print("\n")
            print("SISTEMA DE ECUACIONES NO LINEALES")
            print("\n")
            # Solicitar datos al usuario
            def system(vars):
                x, y = vars
                eq1 = x*2 + y*2 - 1  # Ejemplo de ecuación no lineal
                eq2 = x**3 - y  # Ejemplo de ecuación no lineal
                return [eq1, eq2]

            x0 = float(input("Ingrese una estimación inicial para x: "))
            y0 = float(input("Ingrese una estimación inicial para y: "))
            initial_guess = [x0, y0]

            # Resolver el sistema de ecuaciones no lineales
            solution = fsolve(system, initial_guess)
            x_sol, y_sol = solution

            print(f"Solución encontrada: x = {x_sol}, y = {y_sol}")

            # Crear una malla de puntos para graficar las ecuaciones
            x = np.linspace(-2, 2, 400)
            y = np.linspace(-2, 2, 400)
            X, Y = np.meshgrid(x, y)
            Z1 = X*2 + Y*2 - 1
            Z2 = X**3 - Y

            # Gráfica
            plt.figure(figsize=(8, 6))
            plt.contour(X, Y, Z1, levels=[0], colors='blue', label='x^2 + y^2 - 1 = 0')
            plt.contour(X, Y, Z2, levels=[0], colors='red', label='x^3 - y = 0')
            plt.plot(x_sol, y_sol, 'go', label='Solución')
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Sistema de ecuaciones no lineales")
            plt.grid()
            plt.show()

        if opcion == '6':
            print("\n")
            print("SISTEMA DE ECUACIONES LINEALES")
            print("\n")
            # Solicitar datos al usuario
            n = 2

            A = np.zeros((n, n))
            b = np.zeros(n)

            print("Ingrese los coeficientes de la matriz A:")
            for i in range(n):
                for j in range(n):
                    A[i, j] = float(input(f"A[{i}][{j}]: "))

            print("Ingrese los términos independientes del vector b:")
            for i in range(n):
                b[i] = float(input(f"b[{i}]: "))

            # Resolver el sistema de ecuaciones lineales
            x = np.linalg.solve(A, b)

            print("Solución del sistema:")
            for i in range(n):
                print(f"x[{i}] = {x[i]}")

            # Gráfica (solo para sistemas de 2 ecuaciones)
            if n == 2:
                x_vals = np.linspace(-10, 10, 400)
                y_vals1 = (b[0] - A[0, 0] * x_vals) / A[0, 1]
                y_vals2 = (b[1] - A[1, 0] * x_vals) / A[1, 1]

                plt.figure(figsize=(8, 6))
                plt.plot(x_vals, y_vals1, label=f"{A[0, 0]}*x + {A[0, 1]}*y = {b[0]}", color="blue")
                plt.plot(x_vals, y_vals2, label=f"{A[1, 0]}*x + {A[1, 1]}*y = {b[1]}", color="red")
                plt.plot(x[0], x[1], 'go', label='Solución')
                plt.legend()
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title("Sistema de ecuaciones lineales")
                plt.grid()
                plt.show()

        if opcion == '7':
            
            print("\n")
            print("ECUACION DIFERENCIAL DE PRIMER ORDEN")
            print("\n")

            # Función para obtener la solución de la ecuación diferencial: y(t) = (b/a) + C * exp(-a * t)
            def y(t, a, b, C):
                return (b / a) + C * np.exp(-a * t)  # Solución general para dy/dt + ay = b

            # Pedir al usuario los valores de la ecuación diferencial
            a = float(input("Ingrese el valor de la constante a: "))
            b = float(input("Ingrese el valor de la constante b: "))
            C = float(input("Ingrese el valor de la constante C: "))

            # Pedir al usuario el intervalo de tiempo (inicio, fin) y el número de puntos
            t_start = float(input("Ingrese el tiempo inicial (t_inicio): "))
            t_end = float(input("Ingrese el tiempo final (t_final): "))
            num_points = int(input("Ingrese el número de puntos: "))

            # Crear el intervalo de tiempo basado en la entrada del usuario
            t_vals = np.linspace(t_start, t_end, num_points)

            # Calcular los valores de y(t) para cada t
            y_vals = y(t_vals, a, b, C)

            # Mostrar el resultado de la ecuación
            print(f"\nLa solución de la ecuación diferencial es: y(t) = {b}/{a} + {C} * e^(-{a}t)")

            # Mostrar el valor de y(t) para el tiempo final (opcional)
            print(f"\nValor de y(t) en t = {t_end}: {y(t_end, a, b, C)}")

            # Graficar la solución
            plt.plot(t_vals, y_vals, label=rf'$y(t) = {b}/{a} + Ce^{{-{a}t}}$', color='b')
            plt.xlabel('Tiempo (t)')
            plt.ylabel('y(t)')
            plt.title(f'Solución de la Ecuación Diferencial: dy/dt + {a}y = {b}')
            plt.grid(True)
            plt.legend()

            # Agregar algunos puntos específicos a la gráfica (ejemplo)
            plt.scatter([t_start, t_end], [y(t_start, a, b, C), y(t_end, a, b, C)], color='r', zorder=5)

            plt.show()



        if opcion == '8':
            print("\n")
            print("ECUACION DIFERENCIAL CON TRANSFORMADA DE LAPLACE")
            print("\n")

            def y(t, C, y0):
                return C * (1 - np.exp(-2 * t)) + y0 * np.exp(-2 * t)

            C = float(input("Ingrese el valor de la constante C: "))
            y0 = float(input("Ingrese el valor de y(0): "))  # Condición inicial
            # Pedir al usuario el intervalo de tiempo (inicio, fin) y el número de puntos
            t_start = float(input("Ingrese el tiempo inicial (t_inicio): "))
            t_end = float(input("Ingrese el tiempo final (t_final): "))
            num_points = int(input("Ingrese el número de puntos: "))
                
            # Crear el intervalo de tiempo basado en la entrada del usuario
            t_vals = np.linspace(t_start, t_end, num_points)
                
            # Calcular los valores de y(t) para cada t
            y_vals = y(t_vals, C, y0)  # Agregar la condición inicial
                
            # Mostrar el resultado de la ecuación
            print(f"\nLa solución de la ecuación diferencial es: y(t) = {C}(1 - e^(-2t)) + y(0)e^(-2t)")   
            # Mostrar el valor de y(t) para el tiempo final (opcional)
            print(f"\nValor de y(t) en t = {t_end}: {y(t_end, C, y0)}")
                
            # Graficar la solución
            plt.plot(t_vals, y_vals, label=r'$y(t) = C(1 - e^{-2t}) + y(0)e^{-2t}$')
            plt.xlabel('Tiempo (t)')
            plt.ylabel('y(t)')
            plt.title('Solución de la Ecuación Diferencial (LAPLACE): dy/dt + 2y = 5')
            plt.grid(True)
            plt.legend()
            plt.show()

        elif opcion == '1':
            print("Ha seleccionado Inicio")
        elif opcion == '2':
            print("Ha seleccionado Tylor")
        elif opcion == '3':
            print("Ha seleccionado El método de Newton")
        elif opcion == '5':
            print("Ha seleccionado Sistemas de ecuaciones no lineales")
        elif opcion == '6':
            print("Ha seleccionado Sistemas de ecuaciones lineales")
        elif opcion == '7':
            print("Ha seleccionado ED Primer orden")
        elif opcion == '8':
            print("Ha seleccionado ED Transformada de LAPLACE")
        else:
            print("Opción no válida")

if __name__ == '__main__':
    main()