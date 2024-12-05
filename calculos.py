import numpy as np

def x(t):
    return (50*np.exp(-t))-(10*np.exp(-6*t))

def y(t):
    return (25*np.exp(-t))+(20*np.exp(-6*t))

print(f"soma: {y(3)+x(3)}")