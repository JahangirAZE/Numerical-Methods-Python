import numpy as np
import matplotlib.pyplot as plt

def modified_secant(f, x, error=1e-10, max_iter=100):
    delta = 1e-6 
    if f(x) == f(delta*x + x):
        print("Modified Secant method is not applicable!")
        return None
    for i in range(max_iter):
        x -= f(x)*delta*x/(f(x+delta*x)-f(x))
        if(abs(f(x)) < error):
            print("Root has been found!")
            return x
    print("Function diverges!")
    return None

x = np.linspace(-2, 2, 100)
f = lambda x: x**5 + x**3 + 3
x = -1
root = modified_secant(f, x)

print(f"Root of the function is {root}")
print(f"Checking the roots => {f(root)}")
