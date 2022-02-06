import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# @title Plotting Functions

def histogram(counts, bins, vlines=(), ax=None, ax_args=None, **kwargs):
    """Plot a step histogram given counts over bins."""
    if ax is None:
        _, ax = plt.subplots()

    # duplicate the first element of `counts` to match bin edges
    counts = np.insert(counts, 0, counts[0])

    ax.fill_between(bins, counts, step="pre", alpha=0.4, **kwargs)  # area shading
    ax.plot(bins, counts, drawstyle="steps", **kwargs)  # lines

    for x in vlines:
        ax.axvline(x, color='r', linestyle='dotted')  # vertical line

    if ax_args is None:
        ax_args = {}

    # heuristically set max y to leave a bit of room
    ymin, ymax = ax_args.get('ylim', [None, None])
    if ymax is None:
        ymax = np.max(counts)
        if ax_args.get('yscale', 'linear') == 'log':
            ymax *= 1.5
        else:
            ymax *= 1.1
            if ymin is None:
                ymin = 0

    if ymax == ymin:
        ymax = None

    ax_args['ylim'] = [ymin, ymax]

    ax.set(**ax_args)
    ax.autoscale(enable=False, axis='x', tight=True)


def plot_neuron_stats(v, spike_times):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    # membrane voltage trace
    ax1.plot(v[0:100])
    ax1.set(xlabel='Time', ylabel='Voltage')
    # plot spike events
    for x in spike_times:
        if x >= 100:
            break
        ax1.axvline(x, color='red')

    # ISI distribution
    if len(spike_times) > 1:
        isi = np.diff(spike_times)
        n_bins = np.arange(isi.min(), isi.max() + 2) - .5
        counts, bins = np.histogram(isi, n_bins)
        vlines = []
        if len(isi) > 0:
            vlines = [np.mean(isi)]
        xmax = max(20, int(bins[-1]) + 5)
        histogram(counts, bins, vlines=vlines, ax=ax2, ax_args={
            'xlabel': 'Inter-spike interval',
            'ylabel': 'Number of intervals',
            'xlim': [0, xmax]
        })
    else:
        ax2.set(xlabel='Inter-spike interval',
                ylabel='Number of intervals')
    plt.show()


def lif_neuron(n_steps=1000, alpha=0.01, rate=10):
    """ Simulate a linear integrate-and-fire neuron.

  Args:
    n_steps (int): The number of time steps to simulate the neuron's activity.
    alpha (float): The input scaling factor
    rate (int): The mean rate of incoming spikes

  """
    # Precompute Poisson samples for speed
    exc = stats.poisson(rate).rvs(n_steps)

    # Initialize voltage and spike storage
    v = np.zeros(n_steps)
    spike_times = []

    # ################################################################################
    # # Students: compute dv, then comment out or remove the next line
    # raise NotImplementedError("Excercise: compute the change in membrane potential")
    # ################################################################################

    # Loop over steps
    for i in range(1, n_steps):

        # Update v
        dv = alpha * exc[i - 1]
        v[i] = v[i - 1] + dv

        # If spike happens, reset voltage and record
        if v[i] > 1:
            spike_times.append(i)
            v[i] = 0

    return v, spike_times


# Set random seed (for reproducibility)
np.random.seed(12)

# Model LIF neuron
v, spike_times = lif_neuron()

# Visualize
plot_neuron_stats(v, spike_times)


def lif_neuron_inh(n_steps=1000, alpha=0.5, beta=0.1, exc_rate=10, inh_rate=10):
    """ Simulate a simplified leaky integrate-and-fire neuron with both excitatory
  and inhibitory inputs.

  Args:
    n_steps (int): The number of time steps to simulate the neuron's activity.
    alpha (float): The input scaling factor
    beta (float): The membrane potential leakage factor
    exc_rate (int): The mean rate of the incoming excitatory spikes
    inh_rate (int): The mean rate of the incoming inhibitory spikes
  """

    # precompute Poisson samples for speed
    exc = stats.poisson(exc_rate).rvs(n_steps)
    inh = stats.poisson(inh_rate).rvs(n_steps)

    v = np.zeros(n_steps)
    spike_times = []

    # ###############################################################################
    # # Students: compute dv, then comment out or remove the next line
    # raise NotImplementedError("Excercise: compute the change in membrane potential")
    # ################################################################################

    for i in range(1, n_steps):

        dv = alpha * (exc[i] - inh[i]) - beta * v[i - 1]

        v[i] = v[i - 1] + dv
        if v[i] > 1:
            spike_times.append(i)
            v[i] = 0

    return v, spike_times


# Set random seed (for reproducibility)
np.random.seed(12)

# Model LIF neuron
v, spike_times = lif_neuron_inh()

# Visualize
plot_neuron_stats(v, spike_times)
