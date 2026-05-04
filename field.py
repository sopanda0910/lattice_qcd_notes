import numpy as np
import matplotlib.pyplot as plt

# --- Field Theory Parameters ---
MASS = 1.0       # Mass of the scalar field (m_phi) [What does this ]
N_SPATIAL = 10   # Number of spatial sites (L)
N_TIME = 20      # Number of temporal sites (T)
A = 0.5          # Lattice spacing
N_CF = 5000      # Reduced samples for faster execution
EPS = 0.5        # Metropolis step size

def get_action(phi):
    """
    Scalar field action S[phi]. 
    This extends Eq (53) to include spatial derivatives.
    """
    # Temporal finite difference (periodic)
    dt_phi = (np.roll(phi, -1, axis=0) - phi) / A
    # Spatial finite difference (periodic)
    dx_phi = (np.roll(phi, -1, axis=1) - phi) / A
    
    # S = sum [ 0.5*(d_mu phi)^2 + 0.5*m^2*phi^2 ]
    kinetic = 0.5 * (dt_phi**2 + dx_phi**2)
    potential = 0.5 * (MASS**2) * (phi**2)
    
    return np.sum(kinetic + potential) * (A**2)

def update_field(phi):
    """Metropolis update for the entire 2D lattice."""
    accepted = 0
    for t in range(N_TIME):
        for x in range(N_SPATIAL):
            old_val = phi[t, x]
            old_S = get_action(phi)
            
            phi[t, x] += np.random.uniform(-EPS, EPS)
            new_S = get_action(phi)
            
            if (new_S > old_S) and (np.exp(-(new_S - old_S)) < np.random.rand()):
                phi[t, x] = old_val # Reject
            else:
                accepted += 1
    return accepted

# --- Simulation ---
phi = np.zeros((N_TIME, N_SPATIAL))
samples = []

print("Starting QFT Metropolis simulation...")
for i in range(N_CF):
    update_field(phi)
    if i % 10 == 0: # Save every 10th config
        samples.append(np.copy(phi))

configs = np.array(samples)

# --- Zero-Momentum Projection ---
# As per Eq (44) in image_aaeb72.png, we sum over the spatial index.
# This extracts the p=0 physics.
gamma_t = np.mean(configs, axis=2) # Shape: (N_Samples, N_TIME)

# --- Two-Point Correlator ---
# G(t) = <Gamma(t) Gamma(0)> as per Eq (45) in image_aaeb72.png
g_t = np.zeros(N_TIME)
for t in range(N_TIME):
    g_t[t] = np.mean(gamma_t * np.roll(gamma_t, -t, axis=1))

# --- Visualization ---
plt.figure(figsize=(8, 5))
plt.semilogy(range(N_TIME), g_t, 'o-', label='Lattice Correlator $G(t)$')
plt.title("Scalar Field Theory: Zero-Momentum Correlator")
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.show()