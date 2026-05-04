import numpy as np
import matplotlib.pyplot as plt

# --- Setup ---
N = 15          # Lattice size
BETA = 1.5      # "Temperature" of the vacuum (Lower = more chaotic)
EPS = 0.4       # Metropolis step size
links = np.random.uniform(0, 2*np.pi, (N, N, 2)) # Random start

def update(links):
    """Metropolis sweep for U(1) Gauge Theory."""
    for x in range(N):
        for y in range(N):
            for mu in range(2):
                old_p = links[x, y, mu]
                # Local action: only need plaquettes touching this link
                # For simplicity in this visualization, we'll use a total sum
                old_S = -BETA * np.cos(get_local_plaq(links, x, y))
                
                links[x, y, mu] += np.random.uniform(-EPS, EPS)
                new_S = -BETA * np.cos(get_local_plaq(links, x, y))
                
                if new_S > old_S and np.exp(-(new_S - old_S)) < np.random.rand():
                    links[x, y, mu] = old_p

def get_local_plaq(l, x, y):
    """Returns the phase sum of the plaquette at (x,y)."""
    return l[x,y,0] + l[(x+1)%N,y,1] - l[x,(y+1)%N,0] - l[x,y,1]

# --- Run Simulation ---
print("Simulating... wait for the 'photons' to organize.")
for _ in range(300):
    update(links)

# --- Intuitive Visualization ---
plt.figure(figsize=(8, 8))
X, Y = np.meshgrid(np.arange(N), np.arange(N))

# We visualize the x-component of the gauge field
U = np.cos(links[:,:,0])
V = np.sin(links[:,:,0])

plt.quiver(X, Y, U, V, links[:,:,0], pivot='mid')
plt.title(f"Visualizing the QED Vacuum (Beta={BETA})")
plt.xlabel("Spatial Dimension X")
plt.ylabel("Spatial Dimension Y")
plt.grid(alpha=0.3)
plt.show()