import vegas
import numpy as np
import matplotlib.pyplot as plt

a = 0.5
N = 8
T = a * N
x_lim = 5

normalizer = ((2*np.pi*a)) ** (-4)

harmonic = lambda x : 0.5 * (x**2)
potential_func = lambda x : 0.5 * (x**4)

def propagator(x_val, n_eval=10000, n_itn=10):
    # The batchintegrand decorator evaluates the integral in arrays rather than single points, much more efficient
    @vegas.batchintegrand
    def batch_int(pts):
        # pts - (batch_size, N-1)
        # All the intermediate points
        batch_size = pts.shape[0]

        # paths - [x_i, ...intermediary points ..., x_f]
        paths = np.empty((batch_size, N + 1))
        # All of the first column
        paths[:, 0] = x_val
        paths[:, 1:-1] = pts.reshape((batch_size, N-1))
        # All of the last column
        paths[:, -1] = x_val

        # Kinetic Energy
        diffs = np.diff(paths, axis=1) # What does axis=1 mean? Is that the diffs across the columns in a given row
        k_e = np.sum(diffs**2, axis=1) / (2*a)

        potential = np.sum(a * harmonic(paths[:, 1:]), axis=1)
        # potential = np.sum(a * potential_func(paths[:, 1:]), axis=1)

        action = k_e + potential
        return np.exp(-action)
    
    # Initializes the Integration
    integ = vegas.Integrator([[-x_lim, x_lim]] * (N-1))

    # Training
    # n_eval sets the parameter of the number of points or evaluations to use
    # n_itn is the number of iterations, and every iteration, Vegas adapts the grid to select the density of the grid based on high values.
    # x // 2 is a floored division (rounds down after dividing)
    integ(batch_int, nitn=n_itn//2, neval = n_eval // 2)

    result = integ(batch_int, nitn = n_itn, neval = n_eval)
    return normalizer * result.mean

x_vals = np.linspace(-2, 2, 15)
x_t = np.linspace(-2, 2, 100)
results = []

for x in x_vals:
    results.append(propagator(x))

theory = lambda x : np.exp(-x**2/2)/(np.pi**(0.25))

plt.figure(figsize=(10, 8))
plt.plot(x_vals, results, 'kx')
plt.plot(x_t, theory(x_t), '--')
plt.xlabel('X values')
plt.ylabel('Normalized Probability Amplitude')
plt.grid(False)
plt.show()