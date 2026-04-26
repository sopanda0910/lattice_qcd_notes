import numpy as np

a = 0.5 # Lattice Spacing
eps = 0.1
N = 8 # Number of lattice points

def S(j, x):
    jp = (j+1)%N
    jm = (j-1)%N
    return a * x[j] ** 2 / 2 + x[j]*(x[j] - x[jp] - x[jm]) / a
    # Calculates action
    pass

def update(x):
    for j in range(N):
        old_x = x[j]
        old_Sj = S(j,x)
        x[j] = x[j] = np.random.uniform(-eps, eps)
        dS = S(j, x) - old_Sj
        if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
            x[j] = old_x