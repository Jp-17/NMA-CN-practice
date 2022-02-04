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


# # #################################################
# # ## TODO for students: fill out code to plot histogram ##
# # # Fill out code and comment or remove the next line
# # raise NotImplementedError("Student exercise: You need to plot histogram")
# # #################################################
#
# # Set random number generator
# np.random.seed(2020)
#
# # Initialize t_range, step_end, n, v_n, i and nbins
# t_range = np.arange(0, t_max, dt)
# step_end = len(t_range)
# n = 10000
# v_n = el * np.ones([n, step_end])
# i = i_mean * (1 + 0.1 * (t_max / dt) ** (0.5) * (2 * np.random.random([n, step_end]) - 1))
# nbins = 32
#
# # Loop over time steps
# for step, t in enumerate(t_range):
#
#     # Skip first iteration
#     if step == 0:
#         continue
#
#     # Compute v_n
#     v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])
#
# # Initialize the figure
# plt.figure()
# plt.ylabel('Frequency')
# plt.xlabel('$V_m$ (V)')
#
# # # Plot a histogram at t_max/10 (add labels and parameters histtype='stepfilled' and linewidth=0)
# # plt.hist(v_n[:, int(step_end/10)], nbins, histtype='stepfilled', linewidth=0, label = 't='+ str(t_max / 10) + 's')
# #
# # # Plot a histogram at t_max (add labels and parameters histtype='stepfilled' and linewidth=0)
# # plt.hist(v_n[:, - 1], nbins, histtype='stepfilled', linewidth=0, label = 't='+ str(t_max) + 's')
#
# # Plot a histogram at t_max/10 (add labels and parameters histtype='stepfilled' and linewidth=0)
# plt.hist(v_n[:, int(step_end / 10)], nbins, label='t=' + str(t_max / 10) + 's')
#
# # Plot a histogram at t_max (add labels and parameters histtype='stepfilled' and linewidth=0)
# plt.hist(v_n[:, - 1], nbins, label='t=' + str(t_max) + 's')
#
# # Add legend
# plt.legend()
# plt.show()
#
# # Set random number generator
# np.random.seed(2020)
#
# # Initialize step_end, t_range, n, v_n and i
# t_range = np.arange(0, t_max, dt)
# step_end = len(t_range)
# n = 500
# v_n = el * np.ones([n, step_end])
# i = i_mean * (1 + 0.1 * (t_max / dt) ** (0.5) * (2 * np.random.random([n, step_end]) - 1))
#
# # Initialize spikes and spikes_n
# spikes = {j: [] for j in range(n)}
# spikes_n = np.zeros([step_end])
#
# # #################################################
# # ## TODO for students: add spikes to LIF neuron ##
# # # Fill out function and remove
# # raise NotImplementedError("Student exercise: add spikes to LIF neuron")
# # #################################################
#
# # Loop over time steps
# for step, t in enumerate(t_range):
#
#     # Skip first iteration
#     if step == 0:
#         continue
#
#     # Compute v_n
#     v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])
#
#     # Loop over simulations
#     for j in range(n):
#
#         # Check if voltage above threshold
#         if v_n[j, step] >= vth:
#             # Reset to reset voltage
#             v_n[j, step] = vr
#
#             # Add this spike time
#             spikes[j] += [t]
#
#             # Add spike count to this step
#             spikes_n[step] += 1
#
# # Collect mean Vm and mean spiking rate
# v_mean = np.mean(v_n, axis=0)
# spikes_mean = spikes_n / n
#
# # Initialize the figure
# plt.figure()
#
# # Plot simulations and sample mean
# ax1 = plt.subplot(3, 1, 1)
# for j in range(n):
#     plt.scatter(t_range, v_n[j], color="k", marker=".", alpha=0.01)
# plt.plot(t_range, v_mean, 'C1', alpha=0.8, linewidth=3)
# plt.ylabel('$V_m$ (V)')
#
# # Plot spikes
# plt.subplot(3, 1, 2, sharex=ax1)
# # for each neuron j: collect spike times and plot them at height j
# for j in range(n):
#     # times = ...
#     plt.scatter(spikes[j], j * np.ones_like(spikes[j]), color="k", marker=".", alpha=0.01)
#
# plt.ylabel('neuron')
#
# # Plot firing rate
# plt.subplot(3, 1, 3, sharex=ax1)
# plt.plot(t_range, spikes_mean)
# plt.xlabel('time (s)')
# plt.ylabel('rate (Hz)')
#
# plt.tight_layout()
# plt.show()


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

# #################################################
# ## TODO for students: make a raster ##
# # Fill out function and remove
# raise NotImplementedError("Student exercise: make a raster ")
# #################################################

# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt) ** (0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n, step_end])

# Loop over time steps
for step, t in enumerate(t_range):

    # Skip first iteration
    if step == 0:
        continue

    # Compute v_n
    v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

    # Initialize boolean numpy array `spiked` with v_n > v_thr
    spiked = (v_n[:, step] >= vth)

    # Set relevant values of v_n to v_reset using spiked
    v_n[spiked, step] = vr

    # Set relevant elements in raster to 1 using spiked
    raster[spiked, step] = 1

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, raster)

# #################################################
# ## TODO for students: add refactory period ##
# # Fill out function and remove
# raise NotImplementedError("Student exercise: add refactory period ")
# #################################################

# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt) ** (0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n, step_end])

# Initialize t_ref and last_spike
t_ref = 0.01
last_spike = -t_ref * np.ones([n])

# Loop over time steps
for step, t in enumerate(t_range):

    # Skip first iteration
    if step == 0:
        continue

    # Compute v_n
    v_n[:, step] = v_n[:, step - 1] + (dt / tau) * (el - v_n[:, step - 1] + r * i[:, step])

    # Initialize boolean numpy array `spiked` with v_n > v_thr
    spiked = (v_n[:, step] >= vth)

    # Set relevant values of v_n to v_reset using spiked
    v_n[spiked, step] = vr

    # Set relevant elements in raster to 1 using spiked
    raster[spiked, step] = 1.

    # Initialize boolean numpy array clamped using last_spike, t and t_ref
    clamped = t - last_spike < t_ref

    # Reset clamped neurons to vr using clamped
    v_n[clamped, step] = vr

    # Update numpy array last_spike with time t for spiking neurons
    last_spike[spiked] = t

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, raster)


def ode_step(v, i, dt):
    """
  Evolves membrane potential by one step of discrete time integration

  Args:
    v (numpy array of floats)
      membrane potential at previous time step of shape (neurons)

    v (numpy array of floats)
      synaptic input at current time step of shape (neurons)

    dt (float)
      time step increment

  Returns:
    v (numpy array of floats)
      membrane potential at current time step of shape (neurons)
  """
    v = v + dt / tau * (el - v + r * i)

    return v


# to_remove solution
def spike_clamp(v, delta_spike):
    """
  Resets membrane potential of neurons if v>= vth
  and clamps to vr if interval of time since last spike < t_ref

  Args:
    v (numpy array of floats)
      membrane potential of shape (neurons)

    delta_spike (numpy array of floats)
      interval of time since last spike of shape (neurons)

  Returns:
    v (numpy array of floats)
      membrane potential of shape (neurons)
    spiked (numpy array of floats)
      boolean array of neurons that spiked  of shape (neurons)
  """

    # Boolean array spiked indexes neurons with v>=vth
    spiked = (v >= vth)
    v[spiked] = vr

    # Boolean array clamped indexes refractory neurons
    clamped = (t_ref > delta_spike)
    v[clamped] = vr

    return v, spiked


# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt) ** (0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n, step_end])

# Initialize t_ref and last_spike
mu = 0.01
sigma = 0.007
t_ref = mu + sigma * np.random.normal(size=n)
t_ref[t_ref < 0] = 0
last_spike = -t_ref * np.ones([n])

# Loop over time steps
for step, t in enumerate(t_range):

    # Skip first iteration
    if step == 0:
        continue

    # Compute v_n
    v_n[:, step] = ode_step(v_n[:, step - 1], i[:, step], dt)

    # Reset membrane potential and clamp
    v_n[:, step], spiked = spike_clamp(v_n[:, step], t - last_spike)

    # Update raster and last_spike
    raster[spiked, step] = 1.
    last_spike[spiked] = t

# Plot multiple realizations of Vm, spikes and mean spike rate
with plt.xkcd():
    plot_all(t_range, v_n, raster)


# Simulation class
class LIFNeurons:
    """
  Keeps track of membrane potential for multiple realizations of LIF neuron,
  and performs single step discrete time integration.
  """

    def __init__(self, n, t_ref_mu=0.01, t_ref_sigma=0.002,
                 tau=20e-3, el=-60e-3, vr=-70e-3, vth=-50e-3, r=100e6):
        # Neuron count
        self.n = n

        # Neuron parameters
        self.tau = tau  # second
        self.el = el  # milivolt
        self.vr = vr  # milivolt
        self.vth = vth  # milivolt
        self.r = r  # ohm

        # Initializes refractory period distribution
        self.t_ref_mu = t_ref_mu
        self.t_ref_sigma = t_ref_sigma
        self.t_ref = self.t_ref_mu + self.t_ref_sigma * np.random.normal(size=self.n)
        self.t_ref[self.t_ref < 0] = 0

        # State variables
        self.v = self.el * np.ones(self.n)
        self.spiked = self.v >= self.vth
        self.last_spike = -self.t_ref * np.ones([self.n])
        self.t = 0.
        self.steps = 0

    def ode_step(self, dt, i):
        # Update running time and steps
        self.t += dt
        self.steps += 1

        # One step of discrete time integration of dt
        self.v = self.v + dt / self.tau * (self.el - self.v + self.r * i)

        # ####################################################
        # ## TODO for students: complete the `ode_step` method
        # # Fill out function and remove
        # raise NotImplementedError("Student exercise: complete the ode_step method")
        # ####################################################

        # Spike and clamp
        self.spiked = self.v > self.vth
        self.v[self.spiked] = self.vr
        self.last_spike[self.spiked] = self.t
        clamped = self.last_spike > self.t - self.t_ref
        self.v[clamped] = self.vr

        self.last_spike[self.spiked] = self.t


# Set random number generator
np.random.seed(2020)

# Initialize step_end, t_range, n, v_n and i
t_range = np.arange(0, t_max, dt)
step_end = len(t_range)
n = 500
v_n = el * np.ones([n, step_end])
i = i_mean * (1 + 0.1 * (t_max / dt) ** (0.5) * (2 * np.random.random([n, step_end]) - 1))

# Initialize binary numpy array for raster plot
raster = np.zeros([n, step_end])

# Initialize neurons
neurons = LIFNeurons(n)

# Loop over time steps
for step, t in enumerate(t_range):
    # Call ode_step method
    neurons.ode_step(dt, i[:, step])

    # Log v_n and spike history
    v_n[:, step] = neurons.v
    raster[neurons.spiked, step] = 1.

# Report running time and steps
print(f'Ran for {neurons.t:.3}s in {neurons.steps} steps.')

# Plot multiple realizations of Vm, spikes and mean spike rate
plot_all(t_range, v_n, raster)
