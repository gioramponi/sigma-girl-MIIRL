import pickle
import numpy as np
import argparse
from joblib import Parallel, delayed
import datetime
from algorithms.mlirl import multiple_intention_irl


def discretize_actions(action):
    new_action = np.zeros((action.shape[0], action.shape[1]))
    val = np.zeros(4)
    for s in range(action.shape[0]):
        for t in range(action.shape[1]):
            x, y = action[s][t].clip([-1, -1], [1, 1])
            if 0.5 > x > -0.5 and y > 0:
                new_action[s, t] = 0  # su
                val[0] = 1
            elif 0.5 > x > -0.5 and y <= 0:
                new_action[s, t] = 1  # giu
                val[1] = 1
            elif 0.5 > y > -0.5 and x >= 0:
                new_action[s, t] = 2  # DESTRA
                val[2] = 1
            elif 0.5 > y > -0.5 and x <= 0:
                new_action[s, t] = 3  # SINISTRA
                val[3] = 1
    return new_action


def create_rewards(state_space, goal, grid_size=9):
    rewards = np.zeros((state_space, 3))
    for s in range(state_space):
        x, y = np.floor(s / grid_size), s % grid_size
        if int(x) == goal[0] and int(y) == goal[1]:  # goal state
            rewards[s][2] = 1
        elif 1 <= x <= grid_size - 2 and 1 <= y <= grid_size - 2:  # slow_region
            rewards[s][1] = -1
        else:
            rewards[s][0] = -1
    return rewards


def run(id, seed, args):
    np.random.seed(seed)
    state_space = args.state_dim
    action_space = args.action_dim
    K = args.K
    goal = np.array([8, 4])
    gamma = args.gamma
    n_iterations = args.n_iterations
    rewards = create_rewards(state_space, goal, grid_size=np.sqrt(state_space))
    n_samples_irl = [5, 10, 20, 30, 100]
    res = np.zeros(len(n_samples_irl))
    path = args.load_path
    if args.beta == '':
        betas = [.5]
    else:
        betas = [float(x) for x in args.beta.split(',')]
    for i_beta, beta in enumerate(betas):
        t_s = np.zeros(len(n_samples_irl))
        samp = [id]
        while samp[0] == id:
            samp = np.random.choice(np.arange(10), 1, replace=False)
        for n_i, n_sample in enumerate(n_samples_irl):
            agent_to_data = {}
            for s2 in samp:
                s = id
                agent_to_data['center' + str(s)] = [
                    path + "center/dataset_%s/" % (
                        str(s + 0)), []]
                agent_to_data['up' + str(s)] = [
                    path + "up/dataset_%s/" % (
                        str(s + 0)), []]
                agent_to_data['down' + str(s)] = [
                    path + "down/dataset_%s/" % (
                        str(s2 + 0)), []]
            all_states = []
            all_actions = []
            for agent in agent_to_data:
                _, action, rew, states = pickle.load(open(agent_to_data[agent][0] + 'trajectories.pkl', 'rb'))
                all_actions += list(discretize_actions(action))[:n_sample]
                all_states += list(states)[:n_sample]
            all_states = np.array(all_states)
            all_actions = np.array(all_actions)
            d_start = datetime.datetime.now()
            r = multiple_intention_irl(all_states, all_actions, rewards, K, state_space, action_space, beta,
                                       gamma=gamma,
                                       n_iterations=n_iterations)
            t_s[n_i] = (datetime.datetime.now() - d_start).total_seconds()
            res[n_i] = r
    return res, t_s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', default=2, type=int, help='number of clusters')
    parser.add_argument('--state_dim', type=int, default=81, help='state space dimension')
    parser.add_argument('--action_dim', type=float, default=4, help='action space cardinality')
    parser.add_argument('--beta', type=str, default='.5', help='comma separated valued of beta parmaeter to consider')
    parser.add_argument('--gamma', type=int, default=0.99, help='discount_factor')
    parser.add_argument('--load_path', type=str, default='data/cont_gridworld_multiple/gpomdp/')
    parser.add_argument('--n_jobs', type=int, default=1, help='number of parallel jobs')
    parser.add_argument('--n_iterations', type=int, default=20, help='number of iterations of ml-irl')
    parser.add_argument('--n_experiments', type=int, default=20, help='number of parallel jobs')
    parser.add_argument('--seed', type=int, default=-1, help='random seed, -1 to have a random seed')
    args = parser.parse_args()
    seed = args.seed
    if seed == -1:
        seed = None
    np.random.seed(seed)
    seeds = [np.random.randint(1000000) for _ in range(args.n_experiments)]
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(run)(id, seed, args) for id, seed in zip(range(args.n_experiments), seeds))
    np.save(args.load_path + '/res_mlirl_final13.npy', results)
