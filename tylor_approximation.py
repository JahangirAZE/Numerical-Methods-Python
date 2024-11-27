import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, factorial

x = np.linspace(0, 5, 100)
f = lambda x: np.cos(x**(2/3) / sqrt(2))
y = f(x)

term = 0
for n in range(5):
    term += (-1)**n * (x**(2/3) / sqrt(2))**(2*n) /  factorial(2*n)
    plt.plot(x, term, label = f"order {n}")

def tylor(x):
    term = 0
    for n in range(5):
        term += (-1)**n * (x**(2/3) / sqrt(2))**(2*n) /  factorial(2*n)
    return term

for i in range(100):
    print(f"{x[i]:<10.5f}{term[i]:<10.5f}{y[i]:<10.5f}{abs(term[i] - y[i]):<10.5f}")

plt.plot(x, y, "k", label = "Original function!")
plt.legend()

plt.show()
