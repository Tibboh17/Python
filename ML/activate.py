import numpy as np

# Activation Function
def relu(z):
    return np.maximum(0, z)

# Derivative of Activation Function
def relu_derivative(z):
    return (z > 0).astype(float)