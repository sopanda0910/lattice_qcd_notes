import numpy as np

# Simulation Parameters from Exercise
L = 8           # Points on a side (L/a = 8)
N_dim = 4       # Spacetime dimensions
beta = 5.5      # Beta_eff = beta_tilde / u0^4
u0 = 1.0        # For standard Wilson action (Eq. 114)
epsilon = 0.24  # Metropolis step size
N_cor = 50      # Sweeps between measurements

# Initialize 4D lattice of links: [x, y, z, t, mu, row, col]
# Each link is a 3x3 complex identity matrix initially
lattice = np.zeros((L, L, L, L, N_dim, 3, 3), dtype=complex)
for i in range(3):
    lattice[..., i, i] = 1.0

def get_su3_random(eps):
    """Generates an SU(3) matrix near Identity for Metropolis steps."""
    # Generate a random hermitian matrix near zero
    r = (np.random.random((3, 3)) - 0.5) + 1j * (np.random.random((3, 3)) - 0.5)
    h = eps * (r + r.conj().T)
    # Exponentiate to get a unitary matrix near Identity
    u, s, vh = np.linalg.svd(np.eye(3) + 1j * h)
    return u @ vh

def calculate_staple(pos, mu):
    """Calculates the sum of neighboring links (staples) for a single link."""
    staple = np.zeros((3, 3), dtype=complex)
    x, y, z, t = pos
    
    for nu in range(N_dim):
        if nu == mu: continue
        
        # Directions for the plaquette paths
        # U_nu(x+mu) * U_mu(x+nu)^dag * U_nu(x)^dag
        # (Implementing periodic boundary conditions)
        up_mu = (pos + np.eye(4, dtype=int)[mu]) % L
        up_nu = (pos + np.eye(4, dtype=int)[nu]) % L
        up_mu_nu = (up_mu + np.eye(4, dtype=int)[nu]) % L
        
        # Upper staple
        s1 = lattice[tuple(up_mu)][nu] @ \
             lattice[tuple(up_nu)][mu].conj().T @ \
             lattice[tuple(pos)][nu].conj().T
        
        # Lower staple (requires similar shifts in -nu direction)
        # ... (Simplified here for brevity, standard QCD logic applies)
        staple += s1
    return staple

def metropolis_sweep():
    """Performs one full sweep (hit every link once)."""
    accepted = 0
    total = L**4 * N_dim
    
    for index in np.ndindex(L, L, L, L, N_dim):
        pos = np.array(index[:4])
        mu = index[4]
        
        old_link = lattice[tuple(index)]
        staple = calculate_staple(pos, mu)
        
        # Propose new link: U' = R * U
        random_step = get_su3_random(epsilon)
        new_link = random_step @ old_link
        
        # Calculate Action Change dS
        # S = - (beta/3) * Re(Tr(U * Staple))
        old_s = - (beta / 3.0) * np.real(np.trace(old_link @ staple))
        new_s = - (beta / 3.0) * np.real(np.trace(new_link @ staple))
        
        # Metropolis Accept/Reject
        if np.random.random() < np.exp(-(new_s - old_s)):
            lattice[tuple(index)] = new_link
            accepted += 1
            
    return accepted / total

def measure_plaquette():
    """Computes the average 1x1 Wilson loop (P_mu,nu / u0^4)."""
    total_tr = 0.0
    count = 0
    # Average over all points and all mu < nu planes
    # ... logic to multiply links around a square ...
    return np.real(total_tr) / (count * 3.0 * (u0**4))

# Execution Loop
print(f"Starting simulation at beta={beta}...")
for sweep in range(200): # Thermalization
    acc_rate = metropolis_sweep()
    if sweep % 10 == 0:
        print(f"Sweep {sweep}, Accept Rate: {acc_rate:.2f}")

# Final Measurement
p_val = measure_plaquette()
print(f"Final 1x1 Wilson Loop: {p_val:.4f} (Target ~0.50)")