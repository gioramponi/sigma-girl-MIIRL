import os
import sys
import numpy as np
from tqdm import trange
import time
import glob
import re
import argparse
from trajectories_reader import read_trajectories

path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from sklearn.linear_model import LinearRegression

NUM_ACTIONS = 3
EPISODE_LENGTH = 400
GAMMA = 0.99


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


from sklearn.linear_model import LogisticRegression


def scirl(states, actions, mask, reward_features, gamma):
    # Train the classifier
    mask = mask.astype(bool)
    discounter = gamma ** np.arange(states.shape[1])
    feature_expectation = np.cumsum(reward_features * discounter[None, :, None] * mask[:, :, None], axis=1)

    Xc = feature_expectation[mask, :].reshape(-1, feature_expectation.shape[-1])
    yc = actions[mask].ravel()

    visited_actions = np.unique(actions)

    heuristic_Xc = []
    heuristic_yc = []
    for x, y in zip(Xc, yc):
        for a in visited_actions:
            if a != y:
                heuristic_Xc.append(gamma * x)
                heuristic_yc.append(a)

    heuristic_Xc = np.array(heuristic_Xc)

    Xc = np.vstack((Xc, heuristic_Xc))
    yc = np.array([1] * len(yc) + [0] * len(heuristic_yc))

    shuffler = np.arange(len(Xc), dtype=int)
    np.random.shuffle(shuffler)

    Xc = Xc[shuffler]
    yc = yc[shuffler]

    policy_classifier = LogisticRegression(fit_intercept=False)
    policy_classifier.fit(Xc, yc)

    w = policy_classifier.coef_[0]
    w = w / np.linalg.norm(w, ord=1)

    return w


def csi(states, actions, mask, reward_features, gamma, use_heuristic=False):
    # Train the classifier
    mask = mask.astype(bool)
    n_steps = int(np.sum(mask))
    lens = np.sum(mask, axis=1).astype(int)
    dones = np.zeros(n_steps)
    dones[np.cumsum(lens) - 1] = 1

    Xc = states[mask, :].reshape(-1, states.shape[-1])
    yc = actions[mask].ravel()

    policy_classifier = LogisticRegression(fit_intercept=False, multi_class='multinomial', solver='lbfgs')
    policy_classifier.fit(Xc, yc)

    class2index = dict(zip(policy_classifier.classes_, range(len(policy_classifier.classes_))))

    def q_function(s, a):
        scores = np.dot(policy_classifier.coef_, s)
        return scores[class2index[a]]

    qs = [q_function(s, a) for s, a in zip(Xc, yc)]
    qs_next = [0. if done else q for done, q in zip(dones, np.concatenate((qs[1:], [0.])).tolist())]

    qs = np.array(qs)
    qs_next = np.array(qs_next)

    r_hat = qs - gamma * qs_next
    r_min = np.min(r_hat) - 1

    Xr = reward_features[mask, :].reshape(-1, reward_features.shape[-1])
    yr = r_hat

    if use_heuristic:
        visited_actions = np.unique(actions)

        heuristic_Xr = []
        heuristic_yr = []
        for x, y in zip(Xr, yr):
            for a in visited_actions:
                if a != y:
                    heuristic_Xr.append(gamma * x)
                    heuristic_yr.append(r_min)

        heuristic_Xr = np.array(heuristic_Xr)
        heuristic_yr = np.array(heuristic_yr)

        Xr = np.vstack((Xr, heuristic_Xr))
        yr = np.hstack((yr, heuristic_yr))

    shuffler = np.arange(len(Xr), dtype=int)
    np.random.shuffle(shuffler)

    Xr = Xr[shuffler]
    yr = yr[shuffler]

    reward_regressor = LinearRegression(fit_intercept=False)
    reward_regressor.fit(Xr, yr)

    w = reward_regressor.coef_
    w = w / np.linalg.norm(w, ord=1)

    return w
