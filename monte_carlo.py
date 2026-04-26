import numpy as np

N = 20
a = 0.5
eps = 1.4
N_cor = 20
N_cf = 1000 # 100, 1000, 10000

def S(j, x):
    # Calculates action
    jp = (j+1)%N
    jm = (j-1)%N
    return a * x[j] ** 2 / 2 + x[j]*(x[j] - x[jp] - x[jm]) / a

def metropolis_update(x):
    for j in range(N):
        old_x = x[j]
        old_Sj = S(j,x)
        x[j] = x[j] + np.random.uniform(-eps, eps)
        dS = S(j, x) - old_Sj
        if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
            x[j] = old_x
    
x_0 = np.random.uniform(-1, 1, N)

paths = np.empty((N_cf, N))

for _ in range(5*N_cor):
    metropolis_update(x_0)

paths[0,:] = x_0

for i in range(N_cf-1):
    for _ in range(N_cor):
        metropolis_update(x_0)
    paths[i+1] = x_0.copy()

def get_Gn(n):
    shfited_paths = np.roll(paths, shift=-n, axis=1) # Shifts all the lattice sites to the left by n
    return np.mean(paths*shfited_paths) # Automatically sums over all of the j values for a given path

# Next exercise to find G_n for x^3(t_j + t)*x^3(t_j)
def get_Gn_cubed(n):
    shfited_paths = np.roll(paths, shift=-n, axis=1)
    return np.mean(paths**3 * shfited_paths**3)

delta_E = (1/a)*np.log(get_Gn(1)/get_Gn(2))
delta_E_cubed = (1/a)*np.log(get_Gn_cubed(4)/get_Gn_cubed(5))
print(delta_E)
print(delta_E_cubed)