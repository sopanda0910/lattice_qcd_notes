import numpy as np
import matplotlib.pyplot as plt

# --- Physics Parameters ---
M = 1.0          # Mass
OMEGA = 1.0      # Frequency
N = 20           # Number of lattice sites
A = 0.5          # Lattice spacing
N_CORR = 10      # Correlation length (steps between samples)
N_CF = 10000    # High-statistics samples (10^5)

def potential(x):
    return 0.5 * M * (OMEGA**2) * (x**2)

def delta2(x):
    """Discrete second derivative with periodic boundary conditions."""
    # Essentially, this is the taylor series of the second derivative
    # This roll() shifts all x values to the left (or right) and then does this entire sum
    
    return (np.roll(x, -1) - 2*x + np.roll(x, 1)) / (A**2)

def action(x, improved=False):
    """Calculates the Action S[x] as defined in Eq. (53) and (55)."""
    d2x = delta2(x)
    if improved:
        # Eq (55): Corrected second derivative term
        d2_sq_x = delta2(d2x)
        kinetic = -0.5 * M * np.sum(x * (d2x - (A**2 / 12.0) * d2_sq_x))
    else:
        # Standard kinetic term
        kinetic = -0.5 * M * np.sum(x * d2x)
    
    return A * (kinetic + np.sum(potential(x)))

def metropolis(n_samples, improved=False):
    """The Metropolis Algorithm implementation."""
    x = np.zeros(N)  # Cold start (all zeros)
    samples = []
    accepted = 0
    
    # Thermalization
    for _ in range(1000):
        update(x, improved)
        
    for i in range(n_samples * N_CORR):
        accepted += update(x, improved)
        if i % N_CORR == 0:
            samples.append(np.copy(x))
            
    print(f"{'Improved' if improved else 'Standard'} Acceptance Rate: {accepted/(n_samples*N_CORR*N):.2f}")
    return np.array(samples)

def update(x, improved):
    """Local Metropolis update for each lattice site."""
    count = 0
    eps = 0.5 # Step size
    for j in range(N):
        old_val = x[j]
        old_S = action(x, improved)
        
        x[j] += np.random.uniform(-eps, eps)
        new_S = action(x, improved)
        
        delta_S = new_S - old_S
        
        if delta_S > 0 and np.exp(-delta_S) < np.random.uniform(0, 1):
            x[j] = old_val # Reject
        else:
            count += 1 # Accept
    return count

# --- Main Execution ---
print(f"Running high-statistics comparison (N_cf = {N_CF})...")

# 1. Generate Configurations
configs_std = metropolis(N_CF, improved=False)
configs_imp = metropolis(N_CF, improved=True)

# 2. Statistical Test: Correlation Functions G(t) = <x(t)x(0)>
def get_correlator(configs):
    # Averaging over all starting positions j for better statistics
    g = np.zeros(N)
    for t in range(N):
        g[t] = np.mean(configs * np.roll(configs, -t, axis=1))
    return g

g_std = get_correlator(configs_std)
g_imp = get_correlator(configs_imp)

# 3. Extract Effective Mass: m_eff = log(G(t)/G(t+1))
# Note: For large t, this should plateau at the particle mass (m=1.0)
m_eff_std = np.log(g_std[:-1] / g_std[1:])
m_eff_imp = np.log(g_imp[:-1] / g_imp[1:])

# --- Plotting Results ---
plt.figure(figsize=(10, 5))
t_axis = np.arange(len(m_eff_std))

plt.plot(t_axis, m_eff_std, 'o-', label='Standard Action', alpha=0.7)
plt.plot(t_axis, m_eff_imp, 's-', label='Symanzik Improved', color='red')
plt.axhline(y=OMEGA, color='black', linestyle='--', label='Theoretical Mass')

plt.ylim(0.5, 1.5)
plt.xlabel('Lattice Time (t/a)')
plt.ylabel('Effective Mass')
plt.title(f'Comparison of Improved vs Standard Action (a={A})')
plt.legend()
plt.grid(True)
plt.show()