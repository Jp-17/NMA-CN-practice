import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# @title Download Data
import io
import requests

r = requests.get('https://osf.io/sy5xt/download')
if r.status_code != 200:
    print('Could not download data')
else:
    steinmetz_spikes = np.load(io.BytesIO(r.content), allow_pickle=True)['spike_times']


def entropy(pmf):
    """Given a discrete distribution, return the Shannon entropy in bits.

  This is a measure of information in the distribution. For a totally
  deterministic distribution, where samples are always found in the same bin,
  then samples from the distribution give no more information and the entropy
  is 0.

  For now this assumes `pmf` arrives as a well-formed distribution (that is,
  `np.sum(pmf)==1` and `not np.any(pmf < 0)`)

  Args:
    pmf (np.ndarray): The probability mass function for a discrete distribution
      represented as an array of probabilities.
  Returns:
    h (number): The entropy of the distribution in `pmf`.

  """
    # ############################################################################
    # # Exercise for students: compute the entropy of the provided PMF
    # #   1. Exclude the points in the distribution with no mass (where `pmf==0`).
    # #      Hint: this is equivalent to including only the points with `pmf>0`.
    # #   2. Implement the equation for Shannon entropy (in bits).
    # #  When ready to test, comment or remove the next line
    # raise NotImplementedError("Excercise: implement the equation for entropy")
    # ############################################################################

    # reduce to non-zero entries to avoid an error from log2(0)
    pmf = pmf[pmf > 0]

    # implement the equation for Shannon entropy (in bits)
    h = np.sum(-pmf * np.log2(pmf))

    # return the absolute value (avoids getting a -0 result)
    return np.abs(h)


# @title Plotting Functions

def plot_pmf(pmf, isi_range):
    """Plot the probability mass function."""
    ymax = max(0.2, 1.05 * np.max(pmf))
    pmf_ = np.insert(pmf, 0, pmf[0])
    plt.plot(bins, pmf_, drawstyle="steps")
    plt.fill_between(bins, pmf_, step="pre", alpha=0.4)
    plt.title(f"Neuron {neuron_idx}")
    plt.xlabel("Inter-spike interval (s)")
    plt.ylabel("Probability mass")
    plt.xlim(isi_range)
    plt.ylim([0, ymax])
    plt.show()


n_bins = 50  # number of points supporting the distribution
x_range = (0, 1)  # will be subdivided evenly into bins corresponding to points

bins = np.linspace(*x_range, n_bins + 1)  # bin edges

pmf = np.zeros(n_bins)
pmf[len(pmf) // 2] = 1.0  # middle point has all the mass

# Since we already have a PMF, rather than un-binned samples, `plt.hist` is not
# suitable. Instead, we directly plot the PMF as a step function to visualize
# the histogram:
pmf_ = np.insert(pmf, 0, pmf[0])  # this is necessary to align plot steps with bin edges
plt.plot(bins, pmf_, drawstyle="steps")
# `fill_between` provides area shading
plt.fill_between(bins, pmf_, step="pre", alpha=0.4)
plt.xlabel("x")
plt.ylabel("p(x)")
plt.xlim(x_range)
plt.ylim(0, 1)
plt.show()

# @markdown Execute this cell to visualize a PMF with split mass

pmf = np.zeros(n_bins)
pmf[len(pmf) // 3] = 0.5
pmf[2 * len(pmf) // 3] = 0.5

pmf_ = np.insert(pmf, 0, pmf[0])
plt.plot(bins, pmf_, drawstyle="steps")
plt.fill_between(bins, pmf_, step="pre", alpha=0.4)
plt.xlabel("x")
plt.ylabel("p(x)")
plt.xlim(x_range)
plt.ylim(0, 1)
plt.show()

# @markdown Execute this cell to visualize a PMF of uniform distribution

pmf = np.ones(n_bins) / n_bins  # [1/N] * N

pmf_ = np.insert(pmf, 0, pmf[0])
plt.plot(bins, pmf_, drawstyle="steps")
plt.fill_between(bins, pmf_, step="pre", alpha=0.4)
plt.xlabel("x")
plt.ylabel("p(x)")
plt.xlim(x_range)
plt.ylim(0, 1)
plt.show()

# Call entropy function and print result
print(f"{entropy(pmf):.2f} bits")

n_bins = 50
mean_isi = 0.025
isi_range = (0, 0.25)

bins = np.linspace(*isi_range, n_bins + 1)
mean_idx = np.searchsorted(bins, mean_isi)

# 1. all mass concentrated on the ISI mean
pmf_single = np.zeros(n_bins)
pmf_single[mean_idx] = 1.0

# 2. mass uniformly distributed about the ISI mean
pmf_uniform = np.zeros(n_bins)
pmf_uniform[0:2 * mean_idx] = 1 / (2 * mean_idx)

# 3. mass exponentially distributed about the ISI mean
pmf_exp = stats.expon.pdf(bins[1:], scale=mean_isi)
pmf_exp /= np.sum(pmf_exp)

# @title
# @markdown Run this cell to plot the three PMFs
fig, axes = plt.subplots(ncols=3, figsize=(18, 5))

dists = [  # (subplot title, pmf, ylim)
    ("(1) Deterministic", pmf_single, (0, 1.05)),
    ("(1) Uniform", pmf_uniform, (0, 1.05)),
    ("(1) Exponential", pmf_exp, (0, 1.05))]

for ax, (label, pmf_, ylim) in zip(axes, dists):
    pmf_ = np.insert(pmf_, 0, pmf_[0])
    ax.plot(bins, pmf_, drawstyle="steps")
    ax.fill_between(bins, pmf_, step="pre", alpha=0.4)
    ax.set_title(label)
    ax.set_xlabel("Inter-spike interval (s)")
    ax.set_ylabel("Probability mass")
    ax.set_xlim(isi_range)
    ax.set_ylim(ylim)

plt.show()

print(
    f"Deterministic: {entropy(pmf_single):.2f} bits",
    f"Uniform: {entropy(pmf_uniform):.2f} bits",
    f"Exponential: {entropy(pmf_exp):.2f} bits",
    sep="\n",
)


def pmf_from_counts(counts):
    """Given counts, normalize by the total to estimate probabilities."""
    # ###########################################################################
    # # Exercise: Compute the PMF. Remove the next line to test your function
    # raise NotImplementedError("Student excercise: compute the PMF from ISI counts")
    # ###########################################################################

    pmf = counts / np.sum(counts)

    return pmf


# Get neuron index
neuron_idx = 283

# Get counts of ISIs from Steinmetz data
isi = np.diff(steinmetz_spikes[neuron_idx])
bins = np.linspace(*isi_range, n_bins + 1)
counts, _ = np.histogram(isi, bins)

# Compute pmf
pmf = pmf_from_counts(counts)

# Visualize
plot_pmf(pmf, isi_range)

print(f"Entropy for Neuron {neuron_idx}: {entropy(pmf):.2f} bits")
