import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# --- 1. Parameters ---
N = 30
beta = 5.0        # Coupling strength (high beta = smoother vacuum)
dt = 0.1          # Time step
x_range = np.arange(N)
y_range = np.arange(N)
X, Y = np.meshgrid(x_range, y_range)

# Initialize Gauge Links (U_mu)
theta = np.random.uniform(-0.01, 0.01, (N, N, 2))
links = np.exp(1j * theta)

# Electron setup
sigma = 2.0
centers = [[N//3, N//2], [2*N//3, N//2]]

def get_psi(c1, c2):
    psi1 = np.exp(-((X - c1[0])**2 + (Y - c1[1])**2) / (2 * sigma**2))
    psi2 = np.exp(-((X - c2[0])**2 + (Y - c2[1])**2) / (2 * sigma**2))
    return (psi1 + psi2) + 0j

# --- 2. Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
q1 = ax1.quiver(X, Y, links[:,:,0].real, links[:,:,0].imag, color='black', alpha=0.6, scale=30)
ax1.set_title("Gauge Field (Showing Induced B-Field)")

im = ax2.imshow(np.zeros((N, N)), cmap='magma', origin='lower', extent=[0, N-1, 0, N-1], vmin=0, vmax=1)
q2 = ax2.quiver(X, Y, np.ones((N, N)), np.zeros((N, N)), pivot='mid', scale=40, width=0.005)
ax2.set_title("Two Electrons Creating Current")

# --- 3. Dynamic Coupling Logic ---
def update(frame):
    global links, centers
    
    # A. Move the electron centers (Orbit)
    radius = N // 5
    angle = frame * 0.08
    centers[0] = [N//2 + radius * np.cos(angle), N//2 + radius * np.sin(angle)]
    centers[1] = [N//2 - radius * np.cos(angle), N//2 - radius * np.sin(angle)]
    
    # B. Calculate the current-carrying wave function
    psi = get_psi(centers[0], centers[1])
    
    # C. TWO-WAY COUPLING (The "Back-Reaction")
    # We calculate the current J on each link. J ~ Im(psi*_x * U * psi_x+mu)
    # This current 'kicks' the phase of the links in that direction.
    for mu in [0, 1]:
        # Shift psi to get neighbors for the gradient calculation
        psi_neighbor = np.roll(psi, -1, axis=1-mu) 
        
        # Calculate current density J_mu
        # This is the 'source' term that generates the magnetic field
        current = np.imag(np.conj(psi) * links[:,:,mu] * psi_neighbor)
        
        # Update the links based on the current (Ampere's Law)
        # The gauge field responds to the motion of the matter
        links[:,:,mu] *= np.exp(1j * current * dt * 2.0)

    # D. Update Visuals
    q1.set_UVC(links[:,:,0].real, links[:,:,0].imag)
    
    # Couple the electron phase back to the updated links for the plot
    coupled_psi = psi * (links[:,:,0] * links[:,:,1])
    phase = np.angle(coupled_psi)
    
    im.set_data(np.abs(coupled_psi)**2)
    q2.set_UVC(np.cos(phase), np.sin(phase))
    q2.set_array(phase.ravel())
    
    return q1, im, q2

ani = FuncAnimation(fig, update, frames=200, interval=30, blit=True)
plt.show()