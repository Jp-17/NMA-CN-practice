# Imports

import numpy as np
import matplotlib.pyplot as plt

# # @title Figure settings
# import ipywidgets as widgets       # interactive display
# %config InlineBackend.figure_format = 'retina'
# plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/content-creation/main/nma.mplstyle")

# @title Helper functions

t_max = 150e-3  # second
dt = 1e-3  # second
tau = 20e-3  # second
el = -60e-3  # milivolt
vr = -70e-3  # milivolt
vth = -50e-3  # milivolt
r = 100e6  # ohm
i_mean = 25e-11  # ampere

# #################################################
# ## TODO for students: fill out code to plot histogram ##
# # Fill out code and comment or remove the next line
# raise NotImplementedError("Student exercise: You need to plot histogram")
# #################################################

# Set random number generator
np.random.seed(2020)

# Initialize t_range, step_end, n, v_n, i and nbins
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 10000
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt) ** (0.5) * (2 * np.random.random([n, step_end]) - 1))
nbins = 32

# Loop over time steps
for step, t in enumerate(t_range):

    # Skip first iteration
    if step == 0:
        continue

    # Compute v_n
    v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

# Initialize the figure
plt.figure()
plt.ylabel('Frequency')
plt.xlabel('$V_m$ (V)')

# # Plot a histogram at t_max/10 (add labels and parameters histtype='stepfilled' and linewidth=0)
# plt.hist(v_n[:, int(step_end/10)], nbins, histtype='stepfilled', linewidth=0, label = 't='+ str(t_max / 10) + 's')
#
# # Plot a histogram at t_max (add labels and parameters histtype='stepfilled' and linewidth=0)
# plt.hist(v_n[:, - 1], nbins, histtype='stepfilled', linewidth=0, label = 't='+ str(t_max) + 's')

# Plot a histogram at t_max/10 (add labels and parameters histtype='stepfilled' and linewidth=0)
plt.hist(v_n[:, int(step_end / 10)], nbins, label='t=' + str(t_max / 10) + 's')

# Plot a histogram at t_max (add labels and parameters histtype='stepfilled' and linewidth=0)
plt.hist(v_n[:, - 1], nbins, label='t=' + str(t_max) + 's')

# Add legend
plt.legend()
plt.show()

# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt) ** (0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize spikes and spikes_n
spikes = {j: [] for j in range(n)}
spikes_n = np.zeros([step_end])

# #################################################
# ## TODO for students: add spikes to LIF neuron ##
# # Fill out function and remove
# raise NotImplementedError("Student exercise: add spikes to LIF neuron")
# #################################################

# Loop over time steps
for step, t in enumerate(t_range):

    # Skip first iteration
    if step == 0:
        continue

    # Compute v_n
    v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

    # Loop over simulations
    for j in range(n):

        # Check if voltage above threshold
        if v_n[j, step] >= vth:
            # Reset to reset voltage
            v_n[j, step] = vr

            # Add this spike time
            spikes[j] += [t]

            # Add spike count to this step
            spikes_n[step] += 1

# Collect mean Vm and mean spiking rate
v_mean = np.mean(v_n, axis=0)
spikes_mean = spikes_n / n

# Initialize the figure
plt.figure()

# Plot simulations and sample mean
ax1 = plt.subplot(3, 1, 1)
for j in range(n):
    plt.scatter(t_range, v_n[j], color="k", marker=".", alpha=0.01)
plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
plt.ylabel('$V_m$ (V)')

# Plot spikes
plt.subplot(3, 1, 2, sharex=ax1)
# for each neuron j: collect spike times and plot them at height j
for j in range(n):
    # times = ...
    plt.scatter(spikes[j], j * np.ones_like(spikes[j]), color="k", marker=".", alpha=0.01)

plt.ylabel('neuron')

# Plot firing rate
plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(t_range, spikes_mean)
plt.xlabel('time (s)')
plt.ylabel('rate (Hz)')

plt.tight_layout()
plt.show()

def plot_all(t_range, v, raster=None, spikes=None, spikes_mean=None):
    """
  Plots Time evolution for
  (1) multiple realizations of membrane potential
  (2) spikes
  (3) mean spike rate (optional)

  Args:
    t_range (numpy array of floats)
        range of time steps for the plots of shape (time steps)

    v (numpy array of floats)
        membrane potential values of shape (neurons, time steps)

    raster (numpy array of floats)
        spike raster of shape (neurons, time steps)

    spikes (dictionary of lists)
        list with spike times indexed by neuron number

    spikes_mean (numpy array of floats)
        Mean spike rate for spikes as dictionary

  Returns:
    Nothing.
  """

    v_mean = np.mean(v, axis=0)
    fig_w, fig_h = plt.rcParams['figure.figsize']
    plt.figure(figsize=(fig_w, 1.5 * fig_h))

    ax1 = plt.subplot(3, 1, 1)
    for j in range(n):
        plt.scatter(t_range, v[j], color="k", marker=".", alpha=0.01)
    plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
    plt.xticks([])
    plt.ylabel(r'$V_m$ (V)')

    if raster is not None:
        plt.subplot(3, 1, 2)
        spikes_mean = np.mean(raster, axis=0)
        plt.imshow(raster, cmap='Greys', origin='lower', aspect='auto')

    else:
        plt.subplot(3, 1, 2, sharex=ax1)
        for j in range(n):
            times = np.array(spikes[j])
            plt.scatter(times, j * np.ones_like(times), color="C0", marker=".", alpha=0.2)

    plt.xticks([])
    plt.ylabel('neuron')

    if spikes_mean is not None:
        plt.subplot(3, 1, 3, sharex=ax1)
        plt.plot(t_range, spikes_mean)
        plt.xlabel('time (s)')
        plt.ylabel('rate (Hz)')

    plt.tight_layout()
    plt.show()

# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt) ** (0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize spikes and spikes_n
spikes = {j: [] for j in range(n)}
spikes_n = np.zeros([step_end])

# #################################################
# ## TODO for students: use Boolean indexing ##
# # Fill out function and remove
# raise NotImplementedError("Student exercise: using Boolean indexing")
# #################################################

# Loop over time steps
for step, t in enumerate(t_range):

    # Skip first iteration
    if step == 0:
        continue

    # Compute v_n
    v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

    # Initialize boolean numpy array `spiked` with v_n > v_thr
    spiked = v_n[:, step] > vth
    print(spiked)
    print(np.where(spiked))
    print(np.where(spiked)[0])
    # Set relevant values of v_n to resting potential using spiked
    v_n[spiked, step] = vr

    # Collect spike times
    for j in np.where(spiked)[0]:
        print(j)
        spikes[j] += [t]
        spikes_n[step] += 1

# Collect mean spiking rate
spikes_mean = spikes_n / n

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, spikes=spikes, spikes_mean=spikes_mean)

