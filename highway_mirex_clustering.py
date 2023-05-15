import pickle
import numpy as np
import argparse
from joblib import Parallel, delayed
import datetime
from algorithms import mirex
import pandas as pd
import random


# todo creating reward for every possible state in state space ?
def create_rewards(state_space, goal, grid_size=9):
    # ! just do based on ranking? reward = score? which score from all trans in traj? avg, last one?
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

# todo: discrietize, limit max num_colls, num_offroad = 10, speed = 30, num_configs = state_space
# todo: replace with state repr [2, 3,], size (10, 3) if 10 states
# todo: later: deep QN for sepsis


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
    # collisions_bins = np.array([5, 10])
    # offroad_bins = np.array([5, 10])
    for traj in states:
        for state in traj:
            # discretize every feature according to bins
            state[2] = np.digitize(state[2], speed_bins, right=False)    # 0, 1
            # state[1] = np.digitize(state[1], collisions_bins, right=False)  # 0, 1, 2
            # state[2] = np.digitize(state[2], offroad_bins, right=False) # 0, 1, 2

    return states.astype(int)

# todo: add actions
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

# todo: by Chase
def get_traj_preferences(args):
    prefs_df = pd.read_csv(args.load_path + args.prefs_file)
    trajs_safety = prefs_df.groupby("traj")["safety_score"].mean()

    prefs = []

    traj_indices = prefs_df["traj"].unique()    # ? why
    for t1 in traj_indices:
        t2 = random.choice(traj_indices)
        while (t2 == t1):
            t2 = random.choice(traj_indices)
        if trajs_safety[t1] > trajs_safety[t2]:
            prefs += [(t1, t2)]
        else:
            prefs += [(t2, t1)]

    # array of tuples (ti, tj), where ti > tj, ti,tj = traj numbers
    return prefs

# todo: by Chase
# def rank_trajs(args):



def run(id, seed, args):
    np.random.seed(seed)
    K = args.K
    n_iterations = args.n_iterations

    # retrieve trajs data
    all_states, len_trajs, all_actions, gt_intents = get_states_actions_intents(args)
    # discretize continuous state space into 2 bins (to avoid needing a DQN for their Bellman Update)
    discretize_speed(all_states)
    # ? remember why we did this
    all_states[all_states == 0] = -1

    preferences = get_traj_preferences(args)
    d_start = datetime.datetime.now()
    # todo: some accuracy check
    res = mirex.multiple_intention_irl(all_states, all_actions, preferences, len_trajs, args.num_features, K, n_iterations=n_iterations)
    t_s = (datetime.datetime.now() - d_start).total_seconds()
    return res, t_s


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', default=4, type=int, help='number of clusters')
    parser.add_argument('--max_num_trans', type=int, default=40, help='max number of transition states per trajectory')
    parser.add_argument('--num_features', type=int, default=5)
    parser.add_argument('--n_jobs', type=int, default=1, help='number of parallel jobs')
    parser.add_argument('--n_iterations', type=int, default=20, help='number of iterations of ml-irl')
    parser.add_argument('--n_experiments', type=int, default=20, help='number of parallel jobs')
    parser.add_argument('--seed', type=int, default=-1, help='random seed, -1 to have a random seed')
    parser.add_argument('--load_path', type=str, default='data/car_highway/sample_dataset/')
    parser.add_argument('--trajs_file', type=str, default='condensed_binary_highway_data.csv')
    parser.add_argument('--prefs_file', type=str, default='data_safety_rankings.csv')
    args = parser.parse_args()
    seed = args.seed
    if seed == -1:
        seed = None
    np.random.seed(seed)
    seeds = [np.random.randint(1000000) for _ in range(args.n_experiments)]
    results = Parallel(n_jobs=args.n_jobs, backend='loky')(
        delayed(run)(id, seed, args) for id, seed in zip(range(args.n_experiments), seeds))
    np.save(args.load_path + '/res_mlirl_final13.npy', results)

