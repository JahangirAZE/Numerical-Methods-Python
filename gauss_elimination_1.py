import numpy as np
import matplotlib.pyplot as plt

def gauss_elimination(A, B):
    A = np.array(A, float)
    B = np.array(B, float)
    n = len(B)
    x = np.zeros(n)

    #forward elimination
    for i in range (n - 1):
        for j in range (i + 1, n):
            factor = A[j][i] / A[i][i]
            A[j] -= factor*A[i]
            B[j] -= factor*B[i]

    x[-1] = B[-1] / A[-1][-1]
    #back substitution
    for i in range(n - 2, -1, -1):
        Sum = 0
        for j in range(i + 1, n):
            Sum += x[j] * A[i][j]
        x[i] = (B[i] - Sum) / A[i][i]
    return x

A = np.array([[2, 5, 4, 1],
              [1, 3, 2, 1],
              [2, 10, 9, 7],
              [3, 8, 9, 2]
              ])

B = np.array([20, 11, 40, 37])

roots3 = gauss_elimination(A, B)
roots2 = np.linalg.inv(A).dot(B)
roots1 = np.linalg.solve(A, B)

print(f"Root found using solve method: {roots1}")
print(f"Root found using inverse method: {roots2}")
print(f"Root found using gauss elimination method: {roots3}")
