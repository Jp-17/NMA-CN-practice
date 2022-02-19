# Imports

import numpy as np
import matplotlib.pyplot as plt


# @title Plotting Functions

def plot_data(X):
    """
  Plots bivariate data. Includes a plot of each random variable, and a scatter
  plot of their joint activity. The title indicates the sample correlation
  calculated from the data.

  Args:
    X (numpy array of floats) :   Data matrix each column corresponds to a
                                  different random variable

  Returns:
    Nothing.
  """

    fig = plt.figure(figsize=[8, 4])
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(X[:, 0], color='k')
    plt.ylabel('Neuron 1')
    plt.title('Sample var 1: {:.1f}'.format(np.var(X[:, 0])))
    ax1.set_xticklabels([])
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(X[:, 1], color='k')
    plt.xlabel('Sample Number')
    plt.ylabel('Neuron 2')
    plt.title('Sample var 2: {:.1f}'.format(np.var(X[:, 1])))
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.plot(X[:, 0], X[:, 1], '.', markerfacecolor=[.5, .5, .5],
             markeredgewidth=0)
    ax3.axis('equal')
    plt.xlabel('Neuron 1 activity')
    plt.ylabel('Neuron 2 activity')
    plt.title('Sample corr: {:.1f}'.format(np.corrcoef(X[:, 0], X[:, 1])[0, 1]))
    plt.show()


def plot_basis_vectors(X, W):
    """
  Plots bivariate data as well as new basis vectors.

  Args:
    X (numpy array of floats) :   Data matrix each column corresponds to a
                                  different random variable
    W (numpy array of floats) :   Square matrix representing new orthonormal
                                  basis each column represents a basis vector

  Returns:
    Nothing.
  """

    plt.figure(figsize=[4, 4])
    plt.plot(X[:, 0], X[:, 1], '.', color=[.5, .5, .5], label='Data')
    plt.axis('equal')
    plt.xlabel('Neuron 1 activity')
    plt.ylabel('Neuron 2 activity')
    plt.plot([0, W[0, 0]], [0, W[1, 0]], color='r', linewidth=3,
             label='Basis vector 1')
    plt.plot([0, W[0, 1]], [0, W[1, 1]], color='b', linewidth=3,
             label='Basis vector 2')
    plt.legend()
    plt.show()


def plot_data_new_basis(Y):
    """
  Plots bivariate data after transformation to new bases.
  Similar to plot_data but with colors corresponding to projections onto
  basis 1 (red) and basis 2 (blue). The title indicates the sample correlation
  calculated from the data.

  Note that samples are re-sorted in ascending order for the first
  random variable.

  Args:
    Y (numpy array of floats):   Data matrix in new basis each column
                                 corresponds to a different random variable

  Returns:
    Nothing.
  """
    fig = plt.figure(figsize=[8, 4])
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(Y[:, 0], 'r')
    plt.xlabel
    plt.ylabel('Projection \n basis vector 1')
    plt.title('Sample var 1: {:.1f}'.format(np.var(Y[:, 0])))
    ax1.set_xticklabels([])
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(Y[:, 1], 'b')
    plt.xlabel('Sample number')
    plt.ylabel('Projection \n basis vector 2')
    plt.title('Sample var 2: {:.1f}'.format(np.var(Y[:, 1])))
    ax3 = fig.add_subplot(gs[:, 1])
    ax3.plot(Y[:, 0], Y[:, 1], '.', color=[.5, .5, .5])
    ax3.axis('equal')
    plt.xlabel('Projection basis vector 1')
    plt.ylabel('Projection basis vector 2')
    plt.title('Sample corr: {:.1f}'.format(np.corrcoef(Y[:, 0], Y[:, 1])[0, 1]))
    plt.show()
