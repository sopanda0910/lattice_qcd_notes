import numpy as np
import matplotlib.pyplot as plt

'''
Naive Implementation of the Path Integral
Quantum Oscillator
V(x)=x^2/2

Bounds on x-values are [-5, 5] instead of (-inf, inf)
L = 10

The integral can be evaluated as the L^(N-1)*<e^-S>, where <> is average value
'''

V = lambda x : x**2 / 2
L = 10
m=1
a = 1/2
N = 5

A = (m/(2*np.pi*a))**(N/2)

def action_calc(path):
    # path is shape (Samples, N+1)
    # Kinetic: sum (x_j+1 - x_j)^2
    diffs = np.diff(path, axis=1)
    kinetic = (m / (2 * a)) * np.sum(diffs**2, axis=1)
    
    # Potential: a * sum(V(x_j)) for j=0 to N-1
    potential = a * np.sum(V(path[:, :-1]), axis=1)
    
    return kinetic + potential

def get_propagator(x_0, samples=1000000):
    # Create a block of random paths: (samples, N+1)
    # Intermediate points x[1]...x[N-1] are random
    paths = np.random.uniform(-L/2, L/2, size=(samples, N+1))
    
    # Fix the endpoints
    paths[:, 0] = x_0
    paths[:, -1] = x_0
    
    # Calculate weights e^-S
    weights = np.exp(-action_calc(paths))
    
    # Average weight * Normalization * Volume
    avg_weight = np.mean(weights)
    return avg_weight * A * (L**(N-1))

x_range = np.arange(0, 2, 0.2)
prop = [get_propagator(x) for x in x_range]

plt.figure(figsize=(10,6))
plt.plot(x_range, prop, 'kx', label="Naive Implementation")
plt.show()