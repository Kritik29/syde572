import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')

X = np.array([x1, x2])
mean_class1 = np.array([4, 7]).T

a = np.matmul(mean_class1, X)

print(a)


