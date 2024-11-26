import numpy as np
import matplotlib.pyplot as plt

def lagrange_inverse(y, x_data, y_data):
    n = len(y_data)
    f = 0
    for i in range (n):
        p = 1
        for j in range (n):
            if i == j: continue
            p *= (y - y_data[j]) / (y_data[i] - y_data[j])
        f += p * x_data[i]
    return f

x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([1, 0.5, 0.3333, 0.25, 0.2, 0.1667, 0.1429])
y_add = 0.3
x_add = lagrange_inverse(y_add, x, y)
print(f"The value of x that corresponds to f(x) = 0.3 is {x_add}")

ym = np.linspace(0, 1.2, 100)
xm = lagrange_inverse(ym, x, y)

#plot
plt.plot(x, y, "ro", label = "Original Data")
plt.plot(xm, ym, "k", label = "Lagrange Polynomial")
plt.scatter(x_add, y_add, color = "purple", label = "Additional Point")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Lagrange Inverse Interpolation")
plt.grid()
plt.legend()
plt.show()
