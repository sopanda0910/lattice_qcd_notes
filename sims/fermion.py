import numpy as np
import matplotlib.pyplot as plt

# --- Setup ---
N = 10          # Small lattice for visibility
# A bispinor has 4 components: (Electron Up/Down, Positron Up/Down)
# We'll simulate a slice of the "Electron-Up" component
psi = np.random.normal(0, 1, (N, N)) + 1j * np.random.normal(0, 1, (N, N))

def normalize(field):
    return field / np.sqrt(np.sum(np.abs(field)**2))

psi = normalize(psi)

# --- Intuitive Visualization ---
fig, ax = plt.subplots(figsize=(8, 8))
X, Y = np.meshgrid(np.arange(N), np.arange(N))

# Magnitude = Intensity (Brightness of the particle)
magnitude = np.abs(psi)
# Phase = Direction of the "internal clock"
phase = np.angle(psi)

# Plotting the "Probability Density" as a background
plt.imshow(magnitude**2, cmap='magma', interpolation='bilinear', extent=[0, N-1, 0, N-1])

# Adding "Phase Wheels" (arrows representing the complex phase)
U = np.cos(phase)
V = np.sin(phase)
plt.quiver(X, Y, U, V, phase, cmap='hsv', pivot='mid', scale=20, width=0.005)

plt.title("Bispinor Field: Electron Density & Internal Phase")
plt.xlabel("Spatial Dimension X")
plt.ylabel("Spatial Dimension Y")
plt.colorbar(label="Phase $\\theta$")
plt.show()