import numpy as np
import matplotlib.pyplot as plt

#initial sketch
f = lambda x: np.cos(x) - x*np.exp(x)
df = lambda x: -np.sin(x) - np.exp(x) - x*np.exp(x)
x = np.linspace(-3, 1, 100)
y = f(x)
plt.plot(x, y, "k", label = "cos(x) - x*exp(x)")

#bisection method
def bisection(f, a, b, max_iter = 100, error = 1e-8):
    if f(a) * f(b) > 0:
        return None
    for i in range(max_iter):
        c = (a+b)/2
        if abs(f(c)) < error:
            return c, i
        if f(a) * f(c) > 0:
            a = c
        else: b = c
    return None

#newton-raphson method
def newton_raphson(f, df, x, max_iter = 100, eps = 1e-8):
    if (df(x) == 0): return None
    for i in range (max_iter):
        x -= f(x) / df(x)
        if abs(f(x)) < eps:
            return x, i
    return None

#secant method
def secant(f, x0, x1, max_iter = 100, eps = 1e-8):
    if f(x0) == f(x1): return None
    for i in range(max_iter):
        x = x1 - f(x1) * (x1-x0) / (f(x1)-f(x0))
        if abs(x1 - x0) < eps:
            return x, i
        x0, x1 = x1, x
    return None

#bisection
root_bisection, iter_bisection = bisection(f, -2, -1)
print(f"Root found using bisection method: root={root_bisection}")
print(f"Iterations: {iter_bisection}")
print(f"Check result: {f(root_bisection)}\n")

#newton_raphson
root_newton_raphson, iter_newton_raphson = newton_raphson(f, df, -2)
print(f"Root found using newton-raphson method: root={root_newton_raphson}")
print(f"Iterations: {iter_newton_raphson}")
print(f"Check result: {f(root_newton_raphson)}\n")

#secant
root_secant, iter_secant = secant(f, -2, -2.1)
print(f"Root found using secant method: root={root_secant}")
print(f"Iterations: {iter_secant}")
print(f"Check result: {f(root_secant)}\n")

#plot
plt.plot(root_bisection, f(root_bisection), "rs")
plt.plot(root_newton_raphson, f(root_newton_raphson), "kp")
plt.plot(root_secant, f(root_secant), color = "purple", marker = "^")
plt.legend()
plt.grid()
plt.show()
