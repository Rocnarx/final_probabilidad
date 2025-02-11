import numpy as np
import matplotlib.pyplot as plt
PASOS_GLOBAL = 5

def validar_matriz_transicion(P):
    if not np.allclose(P.sum(axis=1), 1):
        raise ValueError("Cada fila de la matriz de transición debe sumar 1.")

def identificar_estados(P):
    n = P.shape[0]
    transitorios = []
    recurrentes = []
    absorbentes = [i for i in range(n) if P[i, i] == 1 and np.count_nonzero(P[i]) == 1]

    for i in range(n):
        if i in absorbentes:
            recurrentes.append(i)
        else:
            reachable = set()
            stack = [i]
            while stack:
                estado = stack.pop()
                for j in range(n):
                    if P[estado, j] > 0 and j not in reachable:
                        reachable.add(j)
                        stack.append(j)
            
            if any(j in absorbentes for j in reachable):
                transitorios.append(i)
            else:
                recurrentes.append(i)

    return transitorios, recurrentes

def descomponer_matriz(P, transitorios, recurrentes):
    Q = P[np.ix_(transitorios, transitorios)]
    R = P[np.ix_(transitorios, recurrentes)]
    return Q, R

def calcular_matriz_fundamental(Q):
    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)
    return N

def calcular_probabilidades_absorcion(N, R):
    B = N @ R
    return B

def calcular_tiempos_esperados(N):
    return N.sum(axis=1)

def graficar_convergencia(P, pi_estacionaria, pasos=PASOS_GLOBAL):
    estados = np.zeros((pasos, P.shape[0]))
    estados[0] = np.random.dirichlet(np.ones(P.shape[0]))
    
    for t in range(1, pasos):
        estados[t] = estados[t-1] @ P
    
    for i in range(P.shape[0]):
        plt.plot(estados[:, i], label=f"Estado {i+1}")

    for i in range(P.shape[0]):
        plt.hlines(pi_estacionaria[i], 0, pasos, linestyles='dashed', colors='gray')

    plt.title("Convergencia hacia la distribución estacionaria")
    plt.xlabel("Pasos de tiempo")
    plt.ylabel("Probabilidad")
    plt.legend()
    plt.show()

def calcular_distribucion_estacionaria(P):
    P_transpuesta = P.T
    A = P_transpuesta - np.eye(P.shape[0])
    A = np.vstack([A, np.ones(P.shape[0])])
    b = np.append(np.zeros(P.shape[0]), 1)
    
    pi, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    return pi

def main():
    print("Ingrese la matriz de transición (filas separadas por punto y coma, elementos separados por comas):")
    entrada = input()
    try:
        P = np.array([list(map(float, fila.split(','))) for fila in entrada.split(';')])
        validar_matriz_transicion(P)
    except Exception as e:
        print(f"Error en la entrada: {e}")
        return

    transitorios, recurrentes = identificar_estados(P)
    print(f"Estados transitorios: {transitorios}")
    print(f"Estados recurrentes: {recurrentes}")

    if transitorios:
        Q, R = descomponer_matriz(P, transitorios, recurrentes)
        N = calcular_matriz_fundamental(Q)
        B = calcular_probabilidades_absorcion(N, R)
        tiempos = calcular_tiempos_esperados(N)

        print("Matriz fundamental (N):")
        print(N)
        print("Probabilidades de absorción (B):")
        print(B)
        print("Tiempos esperados hasta la absorción:")
        print(tiempos)
    else:
        print("No hay estados transitorios, la cadena es irreducible.")

    pi = calcular_distribucion_estacionaria(P)
    print("Distribución estacionaria:", pi)
    graficar_convergencia(P, pi)

if __name__ == "__main__":
    main()
