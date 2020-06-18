import numpy as np
from utils import  estimate_distribution_params
from joblib import Parallel, delayed
import datetime
import pickle
from run_clustering import em_clustering
import argparse


def run(id, seed, args):
    np.random.seed(seed)
    n_samples_irl = [5, 10, 20, 30, 100]
    res = np.zeros(len(n_samples_irl))
    n_agents = 1
    samp = [id]
    K = args.n_clusters
    t_s = np.zeros(res.shape)
    optimization_iterations = args.optimization_iterations
    path = args.load_path
    for exp in [id]:
        while samp[0] == exp:
            samp = np.random.choice(np.arange(10), n_agents, replace=False)
        for n_i, n_sample in enumerate(n_samples_irl):
            start = datetime.datetime.now()
            agent_to_data = {}
            for s_i in range(1):
                s = exp
                agent_to_data['center'+str(s)] = [
                    path+"center/dataset_%s/" % (
                        str(s+0)), []]
                agent_to_data['up' + str(s)] = [
                    path+"up/dataset_%s/" % (
                        str(s+0)), []]
                agent_to_data['border' + str(s)] = [
                    path+"border/dataset_%s/" % (
                    str(s+0)), []]
            n_agents = 1
            P_true = np.array([[1, 0, 0]*n_agents, [0, 1, 1]*n_agents], dtype=np.float)
            print("Experiment %s" % (exp+1))
            estimated_gradients_all = []

            num_objectives = 3
            for i, agent in enumerate(agent_to_data.keys()):
                read_path = agent_to_data[agent][0]

                estimated_gradients = np.load(read_path + '/gradients/estimated_gradients.npy', allow_pickle=True)
                estimated_gradients = estimated_gradients
                estimated_gradients_all.append(estimated_gradients)
            mus = []
            sigmas = []
            ids = []

            for i, agent in enumerate(list(agent_to_data.keys())):
                    num_episodes, num_parameters, num_objectives = estimated_gradients_all[i].shape[:]
                    mu, sigma = estimate_distribution_params(estimated_gradients=estimated_gradients_all[i][:n_sample],
                                                             other_options=[False, False], cov_estimation=True,
                                                             diag=True)
                    id_matrix = np.identity(num_parameters)
                    mus.append(mu)
                    sigmas.append(sigma)
                    ids.append(id_matrix)

            P, Omega, loss = em_clustering(mus, sigmas, ids, num_clusters=K, num_objectives=num_objectives,
                                           optimization_iterations=optimization_iterations, verbose=False)
            t_s[n_i] = (datetime.datetime.now() - start).total_seconds()
            res[n_i] = max(np.sum(P * P_true), np.sum(P * P_true[::-1]))

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='data/cont_gridworld_multiple/gpomdp/')
    parser.add_argument('--n_jobs', type=int, default=1, help='number of parallel jobs')
    parser.add_argument('--n_clusters', type=int, default=2, help='number of clusters')
    parser.add_argument('--n_experiments', type=int, default=20, help='number of parallel jobs')
    parser.add_argument('--seed', type=int, default=-1, help='random seed, -1 to have a random seed')
    parser.add_argument('--optimization_iterations', type=int, default=1, help='number of clustering iterations')

    args = parser.parse_args()
    seed = args.seed
    if seed == -1:
        seed = None
    np.random.seed(seed)
    seeds = [np.random.randint(1000000) for _ in range(args.n_experiments)]
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(run)(id, seed, args) for id, seed in zip(range(args.n_experiments), seeds))
    pickle.dump(results, open(args.load_path + '/weights_multiple_sigma_girl.npy', 'wb'))

