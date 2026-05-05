import numpy as np
import matplotlib.pyplot as plt

# --- 1. CONFIGURATION & PARAMETERS ---
L = 8
N_DIM = 4
# Parameters for Wilson Action (Eq. 114)
BETA_WILSON = 5.5
U0_WILSON = 1.0

# Parameters for Improved Action (Eq. 103)
BETA_IMP = 1.719
U0_IMP = 0.797

# Simulation settings
EPSILON = 0.24
N_SWEEPS = 100
MEASURE_EVERY = 10
N_SMEAR = 4
ALPHA_SMEAR = 1/12

def get_su3_identity():
    return np.eye(3, dtype=complex)

def project_to_su3(m):
    # The SVD decomposition splits M into U*SIGMA*V, where U and V are unitary
    # Therefore, by just multiplying the unitary matrices, M is converted into a SU(3) matrix
    u, s, vh = np.linalg.svd(m)
    return u @ vh

def get_random_su3(eps):
    r = (np.random.random((3, 3)) - 0.5) + 1j * (np.random.random((3, 3)) - 0.5)
    h = eps * (r + r.conj().T)
    return project_to_su3(np.eye(3) + 1j * h)

# --- 2. CORE PHYSICS ENGINE ---
class LatticeSimulation:
    def __init__(self, beta, u0, improved=False):
        self.beta = beta
        self.u0 = u0
        self.improved = improved
        self.lattice = np.zeros((L, L, L, L, N_DIM, 3, 3), dtype=complex)
        for i in range(3): self.lattice[..., i, i] = 1.0

    def get_link(self, pos, mu):
        return self.lattice[tuple(pos % L)][mu]

    def calculate_staple(self, pos, mu):
        # This essentially calculates the neighbouring values of the U_{\mu} for calculating the plaquettes
        # This makes the process of adjusting based on the action much more efficient
        staple = np.zeros((3, 3), dtype=complex)
        for nu in range(N_DIM):
            if nu == mu: continue
            
            # Upper staple
            pos_mu = pos + np.eye(4, dtype=int)[mu]
            pos_nu = pos + np.eye(4, dtype=int)[nu]
            # The hermitian conjugate is the equivalent to going backwards
            s1 = self.get_link(pos_mu, nu) @ self.get_link(pos_nu, mu).conj().T @ self.get_link(pos, nu).conj().T
            
            # Lower staple
            pos_neg_nu = pos - np.eye(4, dtype=int)[nu]
            pos_mu_neg_nu = pos_mu - np.eye(4, dtype=int)[nu]
            # The hermitian conjugate is the equivalent to going backwards
            s2 = self.get_link(pos_mu_neg_nu, nu).conj().T @ self.get_link(pos_neg_nu, mu).conj().T @ self.get_link(pos_neg_nu, nu)
            
            staple += s1 + s2
        return staple

    def update(self):
        # This is essentially just the metropolis algorithm
        for index in np.ndindex(L, L, L, L, N_DIM):
            pos, mu = np.array(index[:4]), index[4]
            old_u = self.lattice[tuple(index)]
            staple = self.calculate_staple(pos, mu)
            
            new_u = get_random_su3(EPSILON) @ old_u
            
            # Action: S = -(beta/3u0^4) * ReTr(U * Staple)
            coeff = self.beta / (3.0 * (self.u0**4))
            diff = -coeff * np.real(np.trace((new_u - old_u) @ staple))
            
            if np.random.random() < np.exp(-diff):
                self.lattice[tuple(index)] = new_u

    def smear(self):
        # This is primarily to remove high-energy (low wavelength) ultraviolet fluctuations that are unphysical but due to grid spacing
        temp_lattice = np.copy(self.lattice)
        for mu in range(3): # Spatial only
            for index in np.ndindex(L, L, L, L):
                pos = np.array(index)
                staple = np.zeros((3, 3), dtype=complex)
                for nu in range(3):
                    if nu == mu: continue
                    pos_mu = pos + np.eye(4, dtype=int)[mu]
                    pos_nu = pos + np.eye(4, dtype=int)[nu]
                    staple += self.get_link(pos_mu, nu) @ self.get_link(pos_nu, mu).conj().T @ self.get_link(pos, nu).conj().T
                
                # Essentially replacing the U_{\mu} with an average of the matrices in its neighborhood
                comb = (1 - ALPHA_SMEAR) * self.lattice[tuple(pos)][mu] + (ALPHA_SMEAR/4) * staple
                # Makes sure that the U_{\mu} is still a SU(3) matrix
                temp_lattice[tuple(pos)][mu] = project_to_su3(comb)
        self.lattice = temp_lattice
        # This improves convergence

# --- 3. MEASUREMENT & PLOTTING ---
def measure_v_r(sim, r_max=4, t_val=2):
    v_r = []
    # Simplified measurement along X-axis
    for r in range(1, r_max + 1):
        w_t = measure_loop(sim, r, t_val)
        w_t_plus = measure_loop(sim, r, t_val + 1)
        # V(r) = ln(W(r,t)/W(r,t+1))
        v_r.append(np.log(w_t / w_t_plus))
    return v_r

def measure_loop(sim, r, t, mu=0, nu=3):
    """
    Calculates the average W(r, t) for a loop in the mu-nu plane.
    Default: mu=0 (x), nu=3 (t).
    """
    total_tr = 0.0
    count = 0
    
    # Iterate over all possible starting points on the lattice
    for index in np.ndindex(L, L, L, L):
        pos = np.array(index)
        
        # 1. Forward along mu-direction (distance r)
        path = np.eye(3, dtype=complex)
        curr_pos = np.copy(pos)
        for _ in range(r):
            path = path @ sim.get_link(curr_pos, mu)
            curr_pos[mu] += 1
            
        # 2. Forward along nu-direction (distance t)
        for _ in range(t):
            path = path @ sim.get_link(curr_pos, nu)
            curr_pos[nu] += 1
            
        # 3. Backward along mu-direction (distance r)
        for _ in range(r):
            curr_pos[mu] -= 1
            # Moving backward = Hermitian Conjugate
            path = path @ sim.get_link(curr_pos, mu).conj().T
            
        # 4. Backward along nu-direction (distance t)
        for _ in range(t):
            curr_pos[nu] -= 1
            path = path @ sim.get_link(curr_pos, nu).conj().T
            
        total_tr += np.real(np.trace(path))
        count += 1
        
    # Average over volume and normalize by SU(3) dimension (3)
    # Also divide by u0 factors if measuring unrenormalized loops
    return total_tr / (count * 3.0)

# --- RUN SIMULATION ---
results = {}
for name, params in [("Wilson", (BETA_WILSON, U0_WILSON)), ("Improved", (BETA_IMP, U0_IMP))]:
    sim = LatticeSimulation(*params)
    print(f"Running {name} Action...")
    for s in range(N_SWEEPS):
        sim.update()
        if s % MEASURE_EVERY == 0: sim.smear()
    results[name] = measure_v_r(sim)

# --- PLOTTING ---
r_axis = np.arange(1, 5) * 0.25 # Scale by a = 0.25fm
plt.figure(figsize=(8, 5))
for name, data in results.items():
    plt.plot(r_axis, data, 'o-', label=name)

plt.title("Static Quark Potential $V(r)$")
plt.xlabel("r (fm)")
plt.ylabel("a V(r)")
plt.legend()
plt.grid(True)
plt.show()