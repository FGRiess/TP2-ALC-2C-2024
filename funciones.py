import numpy as np

def potenciaDeMatriz(A, n):
    i = 1
    A_n = A
    while i < n:
        A_n = A @ A
    return A_n

def siguientePotencia(A_n, A):
    return A_n @ A

def metodoPotencia(A):
    #Definimos un vector donde guardar los resultados obtenidos para luego obtener el promedio y el desvio estandar
    avals = np.zeros(250)
    #Asumiendo matriz cuadrada, guardamos la dimension
    n = A.shape[0]   
    #Creamos un vector random y aplicamos el metodo de la potencia, repetimos 250 veces
    for i in range(250):
        x_0 = np.random.rand(n)
        x_k = (A @ x_0) / np.linalg.norm(x_0) 
        for j in range(200):
            x_k = (A @ x_k) / np.linalg.norm(x_k)
        avals[i] = np.linalg.norm(x_k)    #Guardamos cada autovalor obtenido
    return np.mean(avals), np.std(avals)  #Retorna el promedio y el desvio estandar, respectivamente 

def solve_LU(L, U, b):   #Función usada en el TP1
    """
    Resuelve el sistema de ecuaciones Ax = b utilizando la descomposición LU.

    Args:
    L (numpy.ndarray): Matriz triangular inferior.
    U (numpy.ndarray): Matriz triangular superior.
    b (numpy.ndarray): Vector del lado derecho del sistema.

    Returns:
    numpy.ndarray: La solución del sistema.
    """
    y = sp.linalg.solve_triangular(L, b, lower=True)
    x = sp.linalg.solve_triangular(U, y)
    return x

def inversaLU(L, U):    #Función usada en el TP1
    """
    Calcula la inversa de una matriz utilizando su descomposición LU.

    Args:
    L (numpy.ndarray): Matriz triangular inferior.
    U (numpy.ndarray): Matriz triangular superior.

    Returns:
    numpy.ndarray: La matriz inversa.
    """
    rows, cols = L.shape
    Inv = np.zeros(L.shape)
    id = np.eye(rows, cols)
    for i in range(rows):
        Inv[:, i] = solve_LU(L, U, id[:, i])

    return Inv  

def calcular_coeficientes_tecnicos(Z, P):    #Funcion usada en el TP1
    """
    Calcula los coeficientes técnicos utilizando la descomposición LU con pivoteo.

    Args:
    Z (numpy.ndarray): Matriz de insumo-producto.
    P (numpy.ndarray): Matriz de producción total.

    Returns:
    numpy.ndarray: Matriz de coeficientes técnicos.
    """
    return Z @ inversaPLU(*calcularPLU(P))