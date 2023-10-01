import numpy as np
import matplotlib.pyplot as plt
import sympy as sym

x1 = sym.Symbol('x1')
x2 = sym.Symbol('x2')

X = np.array([x1, x2])
mean_class1 = np.array([4.09, 3.18]).T
mean_class2 = np.array([2.97, 11.71]).T

a = -1*np.matmul(mean_class1, X) + 0.5*np.matmul(mean_class1, mean_class1.T) 
b = -1*np.matmul(mean_class2, X) + 0.5*np.matmul(mean_class2, mean_class2.T) 

print(a-b)

sol = sym.solve(a-b, x2)

print(sol)

