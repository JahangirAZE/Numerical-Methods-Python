def lagrange (x, x_data, y_data):
    n = len(x_data)
    f = 0
    for i in range (n):
        p = 1
        for j in range (n):
            if i == j: continue
            p *= (x - x_data[j]) / (x_data[i] - x_data[j])
        f += p * y_data[i]
    return f

#data setting
x = np.array([1, 3, 5, 7, 13])
y = np.array([800, 2310, 3090, 3940, 4755])
x_add = 10
y_add = lagrange(x_add, x, y)
print(f"Velocity of the parachutist at t = {x_add} is {y_add} cm/s")

xm = np.linspace(-1, 15, 100)
ym = lagrange(xm, x, y)

#plot
plt.plot(xm, ym, "k", label = "Lagrange Polynomial")
plt.plot(x, y, "ro", label = "Original Data")
plt.scatter(x_add, y_add, color = "purple", label = "Additional Point")

plt.grid()
plt.legend()
plt.show()
