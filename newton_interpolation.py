import numpy as np
import matplotlib.pyplot as plt

def divided_diff(x, y):
    n = len(x)
    coef = np.zeros([n, n])
    coef[:, 0] = y
    for i in range(1, n):
        for j in range(n - i):
            coef[j][i] = (coef[j + 1][i - 1] - coef[j][i - 1]) / (x[j + i] - x[j])
    return coef     

def newton_poly(coef, x_data, x):
    n = len(x_data)
    f = 0
    for i in range (n):
        p = 1
        for j in range (i):
            p *= (x - x_data[j])
        f += p * coef[i]
    return f

#ordering points with regard to their distance from our target (x=8)
x = np.array([5.5, 11, 13, 2, 1, 0, 16, 18])
y = np.array([9.9, 10.2, 9.35, 5.3, 3.134, 0.5, 7.2, 6.2])
x_add = 8
coef = divided_diff(x, y)[0, :]
y_add = newton_poly(coef, x, x_add)
print(f"Value of the function at x = {x_add} is {y_add}")

xm = np.linspace(-1, 19, 100)
ym = newton_poly(coef, x, xm)

# visualizing
plt.plot(xm, ym, "k", label = "Newton Polynomial")
plt.plot(x, y, "ro", label = "Original Data")
plt.scatter(x_add, y_add, color = "Purple", label = "Additional Point")
plt.grid()
plt.legend()
plt.show()
