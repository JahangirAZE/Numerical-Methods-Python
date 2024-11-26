import numpy as np
import matplotlib.pyplot as plt

def tridiagonal(A, B):
    n = len(B)
    upper = np.zeros(n - 1)
    lower = np.zeros(n - 1)
    diagonal = np.zeros(n)
    x = np.zeros(n)
    
    for i in range (n):
        diagonal[i] = A[i][i]
        if i < n - 1:
            upper[i] = A[i][i + 1]
            lower[i] = A[i + 1][i]

    for i in range (1, n):
        f = lower[i - 1] / diagonal[i - 1]
        diagonal[i] -= f*upper[i - 1]
        B[i] -= f*B[i - 1]

    x[-1] = B[-1] / diagonal[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (B[i] - upper[i] * x[i + 1]) / diagonal[i]
    return x

A = [[2.04, -1, 0, 0],
     [-1, 2.04, -1, 0],
     [0, -1, 2.04, -1],
     [0, 0, -1, 2.04]]

B = [40.8, 0.8, 0.8, 200.8]

root1 = np.linalg.solve(A, B)
root2 = tridiagonal(A, B)

print(f"Real roots are: {root1}")
print(f"Roots got by tridiagonal method are: {root2}")
