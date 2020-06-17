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

    df_border = pd.DataFrame()
    df_border['episodes'] = n_samples_irl
    fig, ax = plt.subplots(nrows=1, ncols=1)

    for i, name in enumerate(alg_names):
        if name =='re_irl':
            continue
        ax.plot(n_samples_irl, res_mean[:, i], label=name, marker=mark[i])
        ax.fill_between(n_samples_irl, res_mean[:, i] + res_std[:, i],
                           res_mean[:, i] - res_std[:, i], alpha=0.3)

        df_border[name + '_mean'] = res_mean[:, i]
        df_border[name + '_high'] = res_mean[:, i] + res_std[:, i]
        df_border[name + '_low'] = res_mean[:, i] - res_std[:, i]



    ax.set_xscale('log')
    ax.set_xlabel('Episodes (n)')
    ax.set_ylabel('Performance')
    ax.set_title('EXPERT 1: Go on Border')
    fig.tight_layout()

    ###############################################
    #
    #
    n_runs = all_weights.shape[0]
    res_mean = np.mean(all_weights, axis=0)
    res_std = 2 * np.std(all_weights, axis=0) / np.sqrt(n_runs)
    fig, ax = plt.subplots(nrows=1, ncols=1)

    for i, name in enumerate(alg_names):
        if name =='re_irl':
            continue
        ax.plot(n_samples_irl, res_mean[:, i], label=name, marker=mark[i])
        ax.fill_between(n_samples_irl, res_mean[:, i] + res_std[:, i],
                           res_mean[:, i] - res_std[:, i], alpha=0.3)

        df_border[name + '_w_mean'] = res_mean[:, i]
        df_border[name + '_w_high'] = res_mean[:, i] + res_std[:, i]
        df_border[name + '_w_low'] = res_mean[:, i] - res_std[:, i]


    ax.set_xscale('log')
    ax.set_xlabel('Episodes (n)')
    ax.set_ylabel('Norm difference of weights')
    ax.set_title('EXPERT 1: Go on Border')
    ax.legend()
    fig.tight_layout()
    if save_path is not None:
        df_border.to_csv(save_path + '/gridworld_border.csv', index=None)
    plt.show()


if __name__ == '__main__':
    load_path = 'data/gridworld3'
    with open(load_path + '/agents.pkl', 'rb') as handle:
        alg_names = pickle.load(handle)
    print(alg_names)
    # load returns
    paths = glob.glob(load_path + '/gridworld_res_irl_all_0.1.npy')
    res = np.load(paths[0])  # = np.zeros((10, 10, 2, 3))
    # load weights
    paths = glob.glob(load_path + '/gridworld_res_irl_w_all_0.1.npy')
    all_weights = np.load(paths[0])

    n_samples_irl = [2, 5, 10, 20, 50, 100, 200, 500, 1000,]

    plot_grid(n_samples_irl, res, all_weights, alg_names, load_path)
