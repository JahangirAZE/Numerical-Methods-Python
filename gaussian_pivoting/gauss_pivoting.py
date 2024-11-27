import numpy as np
import matplotlib.pyplot as plt

def partial_pivoting(A, B):
    n = len(B)
    A = np.array(A, float)
    B = np.array(B, float)
    S = [max(abs(i) for i in x) for x in A]
    L = list(range(n))
    x = np.zeros(n)

    for i in range(n-1):
        max_ratio = 0.0
        for j in range(i, n):
            ratio = abs(A[L[j]][i] / S[L[j]])
            if ratio > max_ratio:
                max_ratio = ratio
                pivot = j
        L[pivot], L[i] = L[i], L[pivot]
        for j in range(i+1, n):
            factor = A[L[j]][i] / A[L[i]][i]
            A[L[j]] -= A[L[i]] * factor
            B[L[j]] -= B[L[i]] * factor
            
    x[-1] = B[L[-1]] / A[L[-1]][-1]
    for i in range(n-2, -1, -1):
        summ = 0
        for j in range(i+1, n):
            summ += x[j] * A[L[i]][j]
        x[i] = (B[L[i]] - summ) / A[L[i]][i]
    return x

A = [[1, 1, 1], [6, -4, 5], [5, 2, 2]]
B = [2, 31, 13]

root2 = np.linalg.solve(A, B)
root1 = partial_pivoting(A, B)


print(root1)
print(root2)
