import numpy as np
import matplotlib.pyplot as plt

def newton_system(F, J, x, error = 1e-10, max_iter = 100):
    for i in range(max_iter):
        if abs(np.linalg.norm(F(x))) < error:
            print("Roots have been found!")
            return x
        x -= np.linalg.inv(J(x)).dot(F(x))
    print("Function diverges!")
    return None

F = lambda x: np.array([x[0]**2-x[1]+x[0]*np.cos(np.pi*x[0]),x[1]*x[0]+np.exp(-1*x[0]) - x[0]**(-1)])  
J = lambda x: np.array([[2*x[0]+np.cos(np.pi*x[0])-x[0]*np.pi*np.sin(np.pi*x[0]),-1], [x[1]-np.exp(-1*x[0])+x[0]**(-2), x[0]]])

#first
x = np.linspace(-3, 3, 100)
y = x**2+x*np.cos(np.pi*x)
plt.plot(x, y)

#second
x = np.linspace(-3, 3, 100)
y = (x**-1 - np.exp(-1*x))/x
plt.plot(x, y)

guesses = [0.5, 0.5]
roots = newton_system(F, J, guesses)
root1 = roots[0]; root2 = roots[1]
plt.plot(root1, root2, "rs")

print(f"Root 1 is {root1}\nRoot 2 is {root2}")
print(f"Checking Roots: {F(roots)}")

plt.grid()
plt.show()
