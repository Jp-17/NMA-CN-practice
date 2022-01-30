# Imports

import numpy as np
import matplotlib.pyplot as plt

t_max = 150e-3  # second
dt = 1e-3  # second
tau = 20e-3  # second
el = -60e-3  # milivolt
vr = -70e-3  # milivolt
vth = -50e-3  # milivolt
r = 100e6  # ohm
i_mean = 25e-11  # ampere

print(t_max, dt, tau, el, vr, vth, r, i_mean)

# Loop for 10 steps, variable 'step' takes values from 0 to 9
for step in range(10):
    # Compute value of t
    t = step * dt

    # Compute value of i at this time step
    i = i_mean * (1 + np.sin(2 * np.pi * t / 0.01))

    # Print value of i
    print(f'{t:.3f}', f'{i:.4e}')

# #################################################
# ## TODO for students: fill out compute v code ##
# # Fill out code and comment or remove the next line
# raise NotImplementedError("Student exercise: You need to fill out code to compute v")
# #################################################

# Initialize step_end and v0
step_end = 10
v = el

# Loop for step_end steps
for step in range(step_end):
    # Compute value of t
    t = step * dt

    # Compute value of i at this time step
    i = i_mean * (1 + np.sin((t * 2 * np.pi) / 0.01))

    # Compute v
    v = v + (el - v + r * i) * dt / tau

    # Print value of t and v
    print(f"{t:.3f} {v:.4e}")
