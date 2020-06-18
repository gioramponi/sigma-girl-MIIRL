import matplotlib.pyplot as plt
import glob
import numpy as np
import pandas as pd
import pickle


def plot_grid(n_samples_irl, res, all_weights, alg_names, save_path=None, fix_re_irl=False):
    mark = ['o', 'x', 's', '*', 'o', 'o']
    n_runs = res.shape[0]
    res_mean = np.mean(res, axis=0)
    res_std = 2 * np.std(res, axis=0) / np.sqrt(n_runs)

    df_border = pd.DataFrame()
    df_border['episodes'] = n_samples_irl
    fig, ax = plt.subplots(nrows=1, ncols=1)

    for i, name in enumerate(alg_names):
        mean_to_plot = res_mean[:, i]
        std_to_plot = res_std[:, i]
        if name == 're_irl' and fix_re_irl:
            paths = glob.glob('data/gridworld_reirl/gridworld_res_irl_all_0.1.npy')
            res = np.load(paths[0])  # = np.zeros((10, 10, 2, 3))
            mean_to_plot = np.mean(res, axis=0)[:, 0]
            std_to_plot = 2 * np.std(res, axis=0)[:, 0] / np.sqrt(n_runs)
        ax.plot(n_samples_irl, mean_to_plot, label=name, marker=mark[i])
        ax.fill_between(n_samples_irl, mean_to_plot + std_to_plot,
                           mean_to_plot - std_to_plot, alpha=0.3)

        df_border[name + '_mean'] = mean_to_plot
        df_border[name + '_high'] = mean_to_plot + std_to_plot
        df_border[name + '_low'] = mean_to_plot - std_to_plot



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
        mean_to_plot = res_mean[:, i]
        std_to_plot = res_std[:, i]
        if name == 're_irl' and fix_re_irl:
            paths = glob.glob('data/gridworld_reirl/gridworld_res_irl_w_all_0.1.npy')
            res = np.load(paths[0])  # = np.zeros((10, 10, 2, 3))
            mean_to_plot = np.mean(res, axis=0)[:, 0]
            std_to_plot = 2 * np.std(res, axis=0)[:, 0] / np.sqrt(n_runs)

        ax.plot(n_samples_irl, mean_to_plot, label=name, marker=mark[i])
        ax.fill_between(n_samples_irl, mean_to_plot + std_to_plot,
                        mean_to_plot - std_to_plot, alpha=0.3)

        df_border[name + '_w_mean'] = mean_to_plot
        df_border[name + '_w_high'] = mean_to_plot + std_to_plot
        df_border[name + '_w_low'] = mean_to_plot - std_to_plot


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
    load_path = 'data/gridworld_original'
    try:
        with open(load_path + '/agents.pkl', 'rb') as handle:
            alg_names = pickle.load(handle)
    except:
        alg_names = ['GIRL', 'RA-GIRL', 'REIRL', 'REIRL-POS']
        alg_names += ['CSI', 'SCIRL']
    print(alg_names)
    # load returns
    # paths = glob.glob(load_path + '/gridworld_res_irl_all_0.0.npy')
    paths = glob.glob(load_path + '/gridworld_res_irl_0.0.npy')

    res = np.load(paths[0])  # = np.zeros((10, 10, 2, 3))
    # load weights
    # paths = glob.glob(load_path + '/gridworld_res_irl_w_all_0.0.npy')
    paths = glob.glob(load_path + '/gridworld_res_irl_w_0.0.npy')

    all_weights = np.load(paths[0])

    n_samples_irl = [2, 5, 10, 20, 50, 100, 200, 500, 1000,]

    plot_grid(n_samples_irl, res, all_weights, alg_names, load_path, fix_re_irl=True)
