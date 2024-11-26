import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#data setting
x = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7])
y = np.array([7, 6, 6, 7, 8, 10, 13, 15, 19, 25, 33, 43, 54])
xm = np.linspace(0, 8, 100)

def least_squares_1 (x, y):
    n = len(x)
    X = np.log(x)
    Y = np.log(y)
    A = np.array([[n, sum(x)],
                  [sum(x), sum(x**2)]
                  ])
    B = np.array([sum(X) + sum(Y), sum(X*x) + sum(Y*x)])
    roots = np.linalg.inv(A).dot(B)
    return roots

def least_squares_2 (x, y):
    n= len(x)
    A = np.array([[sum(x**4), sum(x**3), sum(x**2)],
                  [sum(x**3), sum(x**2), sum(x)],
                  [sum(x**2), sum(x), n]
                  ])
    B = np.array([sum(y*x**2), sum(y*x), sum(y)])
    roots = np.linalg.inv(A).dot(B)
    return roots

def least_squares_3 (x, y):
    n= len(x)
    A = np.array([[sum(x**6), sum(x**5), sum(x**4), sum(x**3)],
                  [sum(x**5), sum(x**4), sum(x**3), sum(x**2)],
                  [sum(x**4), sum(x**3), sum(x**2), sum(x)],
                  [sum(x**3), sum(x**2), sum(x), n]
                  ])
    B = np.array([sum(y*x**3), sum(y*x**2), sum(y*x), sum(y)])
    roots = np.linalg.inv(A).dot(B)
    return roots

f1 = lambda x, a, b : (a*b**x) / x
f2 = lambda x, a, b, c : a*x**2 + b*x + c
f3 = lambda x, a, b, c, d : a*x**3 + b*x**2 + c*x + d



#MODEL 1
print("MODEL 1")
roots = least_squares_1(x, y)
a = roots[0]
b = roots[1]

print(f"Roots by least squares method: a={np.exp(a)} b={np.exp(b)}\n")

roots_curve_fit, _ = curve_fit(f1, x, y)
a_curve = roots_curve_fit[0]
b_curve = roots_curve_fit[1]

print(f"Roots by curve method: a={a_curve} b={b_curve}\n")

ym = f1(xm, np.exp(a), np.exp(b))
plt.plot(xm, ym, "r", label = "(a*b**x) / x")

error1 = sum(y - f1(x, np.exp(a), np.exp(b)))



#MODEL 2
print("MODEL 2")
roots = least_squares_2(x, y)
a = roots[0]
b = roots[1]
c = roots[2]
print(f"Roots by least squares method: a={a}, b={b}, c={c}\n")

roots_curve_fit, _ = curve_fit(f2, x, y)
a_curve = roots_curve_fit[0]
b_curve = roots_curve_fit[1]
c_curve = roots_curve_fit[2]
print(f"Roots by curve fit: a={a_curve}, b={b_curve}, c={c_curve}\n")

ym = f2(xm, a, b, c)
plt.plot(xm, ym, "k", label = "ax^2 + bx + c")

error2 = sum((y - f2(x, a, b, c))**2)



#MODEL 3
print("MODEL 3")
roots = least_squares_3(x, y)
a = roots[0]
b = roots[1]
c = roots[2]
d = roots[3]
print(f"Roots by least squares method: a={a}, b={b}, c={c}, d={d}\n")

roots_curve_fit, _ = curve_fit(f3, x, y)
a_curve = roots_curve_fit[0]
b_curve = roots_curve_fit[1]
c_curve = roots_curve_fit[2]
d_curve = roots_curve_fit[3]
print(f"Roots by curve fit: a={a_curve}, b={b_curve}, c={c_curve}, d={d_curve}\n")

ym = f3(xm, a, b, c, d)
plt.plot(xm, ym, "b", label = "ax^3 + bx^2 + cx + d")

error3 = sum((y - f3(x, a, b, c, d))**2)



#printing errors
print(f"Error of model 1: {error1}")
print(f"Error of model 2: {error2}")
print(f"Error of model 3: {error3}")



#plotting
plt.plot(x, y, "*r", label = "Original data")
plt.grid()
plt.legend()
plt.show()
