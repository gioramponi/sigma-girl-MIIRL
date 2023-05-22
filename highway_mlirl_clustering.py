import pandas as pd
import numpy as np
import argparse
from joblib import Parallel, delayed
import datetime
from algorithms.highway_mlirl import multiple_intention_irl


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


def get_states_actions_intents(args):
    data_df = pd.read_csv(args.load_path + args.trajs_file)

    state_features = ["num_collisions", "num_offroad_visits", "speed",
                      "no_collisions", "no_offroad_visits"]
    grouped_data = data_df.groupby("traj")[state_features]
    traj_arrays = []
    for traj_name, traj_group in grouped_data:
        traj_array = traj_group[state_features].to_numpy()
        traj_arrays.append(traj_array)

    # shape = (40 trajs, 40 max_states/traj, 5 features/state)
    # copy states into new array and keep track of length
    all_states = np.full((len(traj_arrays), args.max_num_trans, args.num_features), np.nan)
    len_trajs = np.zeros(len(traj_arrays))
    for t in range(len(traj_arrays)):
        for s in range(len(traj_arrays[t])):
            all_states[t][s] = traj_arrays[t][s]
        len_trajs[t] = s + 1 # i.e. len(traj_arrays[t])

    actions = data_df.groupby('traj')['action'].apply(lambda x: np.array(x))
    # shape = (40 trajs, 40 max_states/traj) -- 1 action per state
    all_actions = np.full((len(traj_arrays), args.max_num_trans), np.nan)
    for t in range(len(traj_arrays)):
        for s in range(len(traj_arrays[t])):
            all_actions[t][s] = actions[t][s]

    # shape = (40 trajs,) -- 1 intention per traj
    gt_intents = data_df.groupby("traj")['intention'].first()

    return all_states, len_trajs.astype(int), all_actions, gt_intents


def discretize_speed(states):
    speed_bins = np.array([25])
    for traj in states:
        for state in traj:
            # discretize speed according to under 25 and above 25 bins
            state[2] = int(np.digitize(state[2], speed_bins, right=False))    # 0, 1
    return states


def assign_state_indices(states):
    idx_dict = {}
    reshaped_states = states.reshape(-1, states.shape[-1])  # -1, 5
    non_nan_states = reshaped_states[~np.isnan(reshaped_states).any(axis=1)]
    # dont count nan in unique states
    unique_states = np.unique(non_nan_states, axis=0).astype(int)

    for i, s in enumerate(unique_states):
        # if state is non-visited (e.g. has nan num_collisions)
        if not np.isnan(s[0]):
            idx_dict[f"{s[0]}_{s[1]}_{s[2]}_{s[3]}_{s[4]}"] = i

    return idx_dict, unique_states, len(idx_dict)


def run(id, seed, args):
    np.random.seed(seed)
    state_space = args.state_dim
    action_space = args.action_dim
    K = args.K
    goal = np.array([8, 4])
    gamma = args.gamma
    n_iterations = args.n_iterations
    n_samples_irl = [5, 10, 20, 30, 100]
    res = np.zeros(len(n_samples_irl))
    path = args.load_path
    if args.beta == '':
        betas = [.5]
    else:
        betas = [float(x) for x in args.beta.split(',')]
    for i_beta, beta in enumerate(betas):
        t_s = np.zeros(len(n_samples_irl))

    # get trajs data
    all_states, len_trajs, all_actions, gt_intents = get_states_actions_intents(args)
    # discretize continuous state space into 2 bins (to avoid needing a DQN for their Bellman Update)
    discretize_speed(all_states)
    # ? remember why we did this
    all_states[all_states == 0] = -1
    # to be able to map each state to a trans prob, zeta and grad
    states_idx_dict, unique_states, state_space = assign_state_indices(all_states)
    features = unique_states

    d_start = datetime.datetime.now()
    z, theta = multiple_intention_irl(all_states, all_actions, features, K, gt_intents,
                               len_trajs, states_idx_dict, state_space, action_space, beta,
                                gamma=gamma,
                                n_iterations=n_iterations)
    t_s = (datetime.datetime.now() - d_start).total_seconds()
    return z, theta, t_s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', default=4, type=int, help='number of clusters')
    parser.add_argument('--state_dim', type=int, default=81, help='state space dimension')
    parser.add_argument('--action_dim', type=float, default=3, help='action space cardinality')
    parser.add_argument('--max_num_trans', type=int, default=40, help='max number of transition states per trajectory')
    parser.add_argument('--num_features', type=int, default=5)
    parser.add_argument('--beta', type=str, default='.5', help='comma separated valued of beta parmaeter to consider')
    parser.add_argument('--gamma', type=int, default=0.99, help='discount_factor')
    parser.add_argument('--load_path', type=str, default='data/car_highway/sample_dataset/')
    parser.add_argument('--trajs_file', type=str, default='condensed_binary_highway_data.csv')
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
