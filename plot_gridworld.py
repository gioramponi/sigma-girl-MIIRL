import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import pickle


def plot_grid(n_samples_irl, res, all_weights, alg_names, save_path=None):
    mark = ['o', 'x', 's', '*', 'o', 'o']
    n_runs = res.shape[0]
    res_mean = np.mean(res, axis=0)
    res_std = 2 * np.std(res, axis=0) / np.sqrt(n_runs)

    df_center = pd.DataFrame()
    df_border = pd.DataFrame()
    df_center['episodes'] = n_samples_irl
    df_border['episodes'] = n_samples_irl
    fig, ax = plt.subplots(nrows=1, ncols=2)

    for i, name in enumerate(alg_names):
        ax[0].plot(n_samples_irl, res_mean[:, 0, i], label=name, marker=mark[i])
        ax[0].fill_between(n_samples_irl, res_mean[:, 0, i] + res_std[:, 0, i],
                           res_mean[:, 0, i] - res_std[:, 0, i], alpha=0.3)

        df_border[name + '_mean'] = res_mean[:, 0, i]
        df_border[name + '_high'] = res_mean[:, 0, i] + res_std[:, 0, i]
        df_border[name + '_low'] = res_mean[:, 0, i] - res_std[:, 0, i]

        ax[1].plot(n_samples_irl, res_mean[:, 1, i], label=name, marker=mark[i])
        ax[1].fill_between(n_samples_irl, res_mean[:, 1, i] + res_std[:, 1, i],
                           res_mean[:, 1, i] - res_std[:, 1, i], alpha=0.3)

        df_center[name + '_mean'] = res_mean[:, 1, i]
        df_center[name + '_high'] = res_mean[:, 1, i] + res_std[:, 1, i]
        df_center[name + '_low'] = res_mean[:, 1, i] - res_std[:, 1, i]


    ax[0].set_xscale('log')
    ax[0].set_xlabel('Episodes (n)')
    ax[0].set_ylabel('Performance')
    ax[0].set_title('EXPERT 1: Go on Border')

    ax[1].set_xscale('log')
    ax[1].set_xlabel('Episodes (n)')
    ax[1].set_ylabel('Performance')
    ax[1].set_title('EXPERT 2: Go in Center')
    ax[1].legend()

    fig.tight_layout()

    ###############################################
    #
    #

    n_runs = all_weights.shape[0]

    res_mean = np.mean(all_weights, axis=0)
    res_std = 2 * np.std(all_weights, axis=0) / np.sqrt(n_runs)


    fig, ax = plt.subplots(nrows=1, ncols=2)

    for i, name in enumerate(alg_names):
        ax[0].plot(n_samples_irl, res_mean[:, 0, i], label=name, marker=mark[i])
        ax[0].fill_between(n_samples_irl, res_mean[:, 0, i] + res_std[:, 0, i],
                           res_mean[:, 0, i] - res_std[:, 0, i], alpha=0.3)

        df_border[name + '_w_mean'] = res_mean[:, 0, i]
        df_border[name + '_w_high'] = res_mean[:, 0, i] + res_std[:, 0, i]
        df_border[name + '_w_low'] = res_mean[:, 0, i] - res_std[:, 0, i]

        ax[1].plot(n_samples_irl, res_mean[:, 1, i], label=name, marker=mark[i])
        ax[1].fill_between(n_samples_irl, res_mean[:, 1, i] + res_std[:, 1, i],
                           res_mean[:, 1, i] - res_std[:, 1, i], alpha=0.3)

        df_center[name + '_w_mean'] = res_mean[:, 1, i]
        df_center[name + '_w_high'] = res_mean[:, 1, i] + res_std[:, 1, i]
        df_center[name + '_w_low'] = res_mean[:, 1, i] - res_std[:, 1, i]

    ax[0].set_xscale('log')
    ax[0].set_xlabel('Episodes (n)')
    ax[0].set_ylabel('Norm difference of weights')
    ax[0].set_title('EXPERT 1: Go on Border')

    ax[1].set_xscale('log')
    ax[1].set_xlabel('Episodes (n)')
    ax[1].set_ylabel('Norm difference of weights')
    ax[1].set_title('EXPERT 2: Go in Center')
    ax[1].legend()

    fig.tight_layout()
    if save_path is not None:
        df_center.to_csv(save_path + '/gridworld_center.csv', index=None)
        df_border.to_csv(save_path + '/gridworld_border.csv', index=None)
    plt.show()

if __name__ == '__main__':
    load_path = 'data/gridworld'
    with open(load_path + '/agents.pkl', 'rb') as handle:
        alg_names = pickle.load(handle)
    print(alg_names)
    # load returns
    paths = glob.glob('data/gridworld/gridworld_res_irl_all_0.1.npy')
    res = np.load(paths[0])  # = np.zeros((10, 10, 2, 3))
    # load weights
    paths = glob.glob(load_path + '/gridworld_res_irl_w_all_0.1.npy')
    all_weights = np.load(paths[0])

    n_samples_irl = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]

    plot_grid(n_samples_irl, res, all_weights, alg_names, load_path)
