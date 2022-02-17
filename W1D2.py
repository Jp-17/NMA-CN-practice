# Imports

import numpy as np
import matplotlib.pyplot as plt

# for random distributions:
from scipy.stats import norm, poisson

# for logistic regression:
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# @title Plotting Functions

def rasterplot(spikes, movement, trial):
    [movements, trials, neurons, timepoints] = np.shape(spikes)

    trial_spikes = spikes[movement, trial, :, :]

    trial_events = [((trial_spikes[x, :] > 0).nonzero()[0] - 150) / 100 for x in range(neurons)]

    plt.figure()
    dt = 1 / 100
    plt.eventplot(trial_events, linewidths=1);
    plt.title('movement: %d - trial: %d' % (movement, trial))
    plt.ylabel('neuron')
    plt.xlabel('time [s]')


def plotCrossValAccuracies(accuracies):
    f, ax = plt.subplots(figsize=(8, 3))
    ax.boxplot(accuracies, vert=False, widths=.7)
    ax.scatter(accuracies, np.ones(8))
    ax.set(
        xlabel="Accuracy",
        yticks=[],
        title=f"Average test accuracy: {accuracies.mean():.2%}"
    )
    ax.spines["left"].set_visible(False)


# @title Generate Data

def generateSpikeTrains():
    gain = 2
    neurons = 50
    movements = [0, 1, 2]
    repetitions = 800

    np.random.seed(37)

    # set up the basic parameters:
    dt = 1 / 100
    start, stop = -1.5, 1.5
    t = np.arange(start, stop + dt, dt)  # a time interval
    Velocity_sigma = 0.5  # std dev of the velocity profile
    Velocity_Profile = norm.pdf(t, 0, Velocity_sigma) / norm.pdf(0, 0,
                                                                 Velocity_sigma)  # The Gaussian velocity profile, normalized to a peak of 1

    # set up the neuron properties:
    Gains = np.random.rand(neurons) * gain  # random sensitivity between 0 and `gain`
    FRs = (np.random.rand(neurons) * 60) - 10  # random base firing rate between -10 and 50

    # output matrix will have this shape:
    target_shape = [len(movements), repetitions, neurons, len(Velocity_Profile)]

    # build matrix for spikes, first, they depend on the velocity profile:
    Spikes = np.repeat(Velocity_Profile.reshape([1, 1, 1, len(Velocity_Profile)]),
                       len(movements) * repetitions * neurons, axis=2).reshape(target_shape)

    # multiplied by gains:
    S_gains = np.repeat(
        np.repeat(Gains.reshape([1, 1, neurons]), len(movements) * repetitions, axis=1).reshape(target_shape[:3]),
        len(Velocity_Profile)).reshape(target_shape)
    Spikes = Spikes * S_gains

    # and multiplied by the movement:
    S_moves = np.repeat(np.array(movements).reshape([len(movements), 1, 1, 1]),
                        repetitions * neurons * len(Velocity_Profile), axis=3).reshape(target_shape)
    Spikes = Spikes * S_moves

    # on top of a baseline firing rate:
    S_FR = np.repeat(
        np.repeat(FRs.reshape([1, 1, neurons]), len(movements) * repetitions, axis=1).reshape(target_shape[:3]),
        len(Velocity_Profile)).reshape(target_shape)
    Spikes = Spikes + S_FR

    # can not run the poisson random number generator on input lower than 0:
    Spikes = np.where(Spikes < 0, 0, Spikes)

    # so far, these were expected firing rates per second, correct for dt:
    Spikes = poisson.rvs(Spikes * dt)

    return (Spikes)


def subsetPerception(spikes):
    movements = [0, 1, 2]
    split = 400
    subset = 40
    hwin = 3

    [num_movements, repetitions, neurons, timepoints] = np.shape(spikes)

    decision = np.zeros([num_movements, repetitions])

    # ground truth for logistic regression:
    y_train = np.repeat([0, 1, 1], split)
    y_test = np.repeat([0, 1, 1], repetitions - split)

    m_train = np.repeat(movements, split)
    m_test = np.repeat(movements, split)

    # reproduce the time points:
    dt = 1 / 100
    start, stop = -1.5, 1.5
    t = np.arange(start, stop + dt, dt)

    w_idx = list((abs(t) < (hwin * dt)).nonzero()[0])
    w_0 = min(w_idx)
    w_1 = max(w_idx) + 1  # python...

    # get the total spike counts from stationary and movement trials:
    spikes_stat = np.sum(spikes[0, :, :, :], axis=2)
    spikes_move = np.sum(spikes[1:, :, :, :], axis=3)

    train_spikes_stat = spikes_stat[:split, :]
    train_spikes_move = spikes_move[:, :split, :].reshape([-1, neurons])

    test_spikes_stat = spikes_stat[split:, :]
    test_spikes_move = spikes_move[:, split:, :].reshape([-1, neurons])

    # data to use to predict y:
    x_train = np.concatenate((train_spikes_stat, train_spikes_move))
    x_test = np.concatenate((test_spikes_stat, test_spikes_move))

    # this line creates a logistics regression model object, and immediately fits it:
    population_model = LogisticRegression(solver='liblinear', random_state=0).fit(x_train, y_train)

    # solver, one of: 'liblinear', 'newton-cg', 'lbfgs', 'sag', and 'saga'
    # some of those require certain other options
    # print(population_model.coef_)       # slope
    # print(population_model.intercept_)  # intercept

    ground_truth = np.array(population_model.predict(x_test))
    ground_truth = ground_truth.reshape([3, -1])

    output = {}
    output['perception'] = ground_truth
    output['spikes'] = spikes[:, split:, :subset, :]

    return (output)


def getData():
    spikes = generateSpikeTrains()

    dataset = subsetPerception(spikes=spikes)

    return (dataset)


dataset = getData()
perception = dataset['perception']
spikes = dataset['spikes']
