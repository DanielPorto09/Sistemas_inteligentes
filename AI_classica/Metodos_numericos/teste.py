def f(x):
    return (x**3) - (3*x) - 2

def y(x):
    return x**3 , x**2, x, (f(x) +2)

a, b, c, d = y(1)
a_1,b_1,c_1,d_1 = y(1.5)
a_2, b_2,c_2,d_2 = y(3)

print(f"{a}, {b}, {c}, {d}")
print(f"{a_1}, {b_1}, {c_1}, {d_1}")
print(f"{a_2}, {b_2}, {c_2}, {d_2}")