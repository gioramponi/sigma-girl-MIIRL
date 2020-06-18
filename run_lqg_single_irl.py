import os
import sys
import numpy as np
import argparse
import csv
import time
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from envs.lqg import LQG
from policies.linear_gaussian_policy import LinearGaussianPolicy
from utils import compute_gradient
from algorithms.pgirl import solve_ra_PGIRL
from trajectories_reader import read_trajectories
from algorithms.REIRL import RelativeEntropyIRL
from algorithms.CSI import csi


def discretize_actions(actions, action_max, action_min, n_bins_per_dim):
    actions = np.clip(actions, action_min, action_max)
    dims = len(action_max)
    discretized_actions = np.zeros(actions.shape[:2])

    for i in range(dims):
        bins = np.linspace(action_min[i] - 1e-12, action_max[i] + 1e-12, n_bins_per_dim + 1)
        indices = np.digitize(actions[:, :, i].ravel(), bins).reshape(actions[:, :, i].shape) - 1
        assert np.max(indices) <= n_bins_per_dim - 1 and np.min(indices) >= 0
        discretized_actions += indices * n_bins_per_dim ** i
    assert np.min(discretized_actions) >= 0 and np.max(discretized_actions) <= n_bins_per_dim ** dims - 1
    return discretized_actions.astype(int)


def collect_trajectories(demonstrations, env, policy, diff_scales, random_init=False):
    trajectories = []
    demo_features = []
    for i in range(demonstrations):
        trajectory = []
        features = []
        s = env.reset(random_init=random_init, different_scales=diff_scales)
        while True:
            a = policy(s)
            next_s, r, done, info = env.step(a)
            trajectory_sample = {
                's': list(s.reshape((args.dimension,))),
                'a': list(a.reshape((args.dimension,))),
                "s'": list(next_s.reshape((args.dimension,))),
                'r': r,
                'features': list(info['features']),
                'done': done
            }
            features.append(list(info['features']))
            s = next_s
            trajectory.append(trajectory_sample)
            if done:
                trajectories.append(trajectory)
                demo_features.append(np.array(features))
                break

    return trajectories, trajectory_sample.keys(), np.stack(demo_features, axis=0)


parser = argparse.ArgumentParser()
parser.add_argument('--dimension', type=int, default=2, help='number of dimensions of the state and action space')
parser.add_argument("--dir_to_read", type=str, default='', help='directory to read the datasets, if left to default'
                                                                'new demonstrations will be generated')
parser.add_argument('--features_idx', default='', type=str, help='comma separated indexes of the reward features'
                                                                 ' to consider, default consider all')
parser.add_argument('--settings', default='', type=str)
parser.add_argument('--state_noise', type=float, default=0.3, help='noise of the transition kernel')
parser.add_argument('--action_noise', type=float, default=0.3, help='noise in the policy')
parser.add_argument('--diag_state_noise', default='0.0,0.0', type=str, help='vector of coma separated values to specify'
                                                                            ' more complex noise models')
parser.add_argument('--diag_action_noise', default='0.1,2', type=str, help='vector of coma separated values to specify'
                                                                           ' more complex noise models')
parser.add_argument('--a', default='1,0.2,0.2,1', type=str, help='parameter of the reward')
parser.add_argument('--b', default='1,0.2,0.2,1', type=str, help='parameter of the reward')
parser.add_argument('--action_scales', default='0.02, 0.05', type=str, help='Induce non-symmetrical effect'
                                                                            ' of the actions')
parser.add_argument('--max_state', type=float, default=5, help='maximum state')
parser.add_argument('--max_action', type=float, default=np.inf, help='maximum action')
parser.add_argument('--max_rand_action', type=float, default=15, help='parameter for REIRL secondary dataset')
parser.add_argument('--horizon', type=int, default=40, help='horizon of the task')
parser.add_argument('--random_demo', type=int, default=20, help='number of random policy demonstrations')
parser.add_argument('--num_disc', type=int, default=3, help='number of discretized actions')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--seed', type=float, default=1, help='random seed')
parser.add_argument('--shift', type=float, default=0.5, help='make the reward function non-uniform')
parser.add_argument('--num_experiments', type=int, default=10, help='number of experiments to average upon')
parser.add_argument('--verbose', action='store_true', help='enable_logging')
parser.add_argument('--random_init', action='store_true', help='start the episodes in random positions')
parser.add_argument('--random', action='store_true', help='execute random policy')
parser.add_argument('--interactive', action='store_true', help='used when demonstrations is 0,'
                                                               'stop execution at each time step')
parser.add_argument('--print_states', action='store_true', help='log state features')
parser.add_argument('--save_policy', action='store_true', help='save the weights of the optimal policy')
parser.add_argument('--save_gradients', action='store_true', help='save the computed gradients')
parser.add_argument('--load_gradients', action='store_true', help='load the precomputed policy')
parser.add_argument('--diff_scales', action='store_true', help='impact the initial state probabilities')
parser.add_argument('--rotation', action='store_true', help='use a rotation matrix as state transition kernel')
parser.add_argument('--grad_reirl', action='store_true', help='method of reirl solving')
parser.add_argument('--eps_eval', type=int, default=100, help='number of episodes of evaluation for computing J')

args = parser.parse_args()
# parse environment parameters and instantiate the environment
if args.features_idx == '':
    features_idx = list(range(2 * args.dimension))
else:
    features_idx = [int(x) for x in args.features_idx.split(',')]

if args.diag_state_noise == '':
    state_noise = args.state_noise
else:
    state_noise = [float(x) for x in args.diag_state_noise.split(',')]
    if len(state_noise) > args.dimension:
        state_noise = state_noise[:args.dimension]
    state_noise = np.diag(state_noise)

if args.a == '':
    a = np.eye(args.dimension)
else:
    a = [float(x) for x in args.a.split(',')]
    if len(a) > args.dimension ** 2:
        a = a[:args.dimension ** 2]
    a = np.array(a).reshape((args.dimension, args.dimension))

if args.b == '':
    b = np.eye(args.dimension)
else:
    b = [float(x) for x in args.b.split(',')]
    if len(b) > args.dimension ** 2:
        b = b[:args.dimension ** 2]
    b = np.array(b).reshape((args.dimension, args.dimension))

if args.diag_state_noise == '':
    state_noise = args.state_noise
else:
    state_noise = [float(x) for x in args.diag_state_noise.split(',')]
    if len(state_noise) > args.dimension:
        state_noise = state_noise[:args.dimension]
    state_noise = np.diag(state_noise)

if args.action_scales == '':
    action_scales = np.ones(args.dimension)
else:
    action_scales = [float(x) for x in args.action_scales.split(',')]

if args.diag_action_noise == '':
    action_noise = args.action_noise
else:
    action_noise = [float(x) for x in args.diag_action_noise.split(',')]
    if len(action_noise) > args.dimension:
        action_noise = action_noise[:args.dimension]
    action_noise = np.diag(action_noise)

np.random.seed(args.seed)
env = LQG(n=args.dimension,
          horizon=args.horizon,
          gamma=args.gamma,
          max_state=args.max_state,
          max_action=args.max_action,
          sigma_noise=state_noise,
          action_scales=action_scales,
          a=a,
          b=b,
          shift=args.shift)

env_test = LQG(n=args.dimension,
               horizon=args.horizon,
               gamma=args.gamma,
               max_state=args.max_state,
               max_action=args.max_action,
               sigma_noise=state_noise,
               action_scales=action_scales,
               a=a,
               b=b,
               rotation=args.rotation,
               shift=args.shift)


def log(message):
    if args.verbose:
        print(message)


# kpis to record
kpis = ['num_episodes', 'run', 'J_opt_sigma', 'J_opt', 'J_sigma', 'J', 'loss', 'weights', 'diff_weights', 'diff_pi',
        'k']

# settings to run and compare, together with their specific paramenters
setting_to_reqs = {
    'pgirl': (solve_ra_PGIRL, {'seed': args.seed, 'girl': True, 'other_options': [False, True, False]}, []),
    'ra_pgirl_diag': (solve_ra_PGIRL, {"diag": True, 'seed': args.seed}, []),
    #'ra_pgirl_full': (solve_ra_PGIRL, {'seed': args.seed}, []),
    #'ra_pgirl_convex': (solve_ra_PGIRL, {'seed': args.seed, 'other_options': [False, False, False, True]}, []),
    're_irl': (RelativeEntropyIRL, {}, []),
    'csi': (csi, {}, []),
    'ra_pgirl_cov_estimation': (solve_ra_PGIRL, {"cov_estimation": True, 'seed': args.seed}, []),
    # 'ra_pgirl_identity': (solve_ra_PGIRL, {"identity": True, 'seed': args.seed}, []),
}
read_opt = {}

settings = setting_to_reqs.keys()
if args.settings != '':
    settings = [x for x in args.settings.split(',')]
# compute optimal lqg policy in closed form
K = env.computeOptimalK()
Sigma = state_noise
opt_weights = env.get_rew_weights()
opt_J_sigma = env.computeJ(K, Sigma, args.eps_eval, random_init=False)
opt_J = env.computeJ(K, np.diag(np.zeros(args.dimension)), args.eps_eval, random_init=False)

pi = LinearGaussianPolicy()
random_pi = LinearGaussianPolicy(weights=np.zeros(shape=(args.dimension, args.dimension)),
                                 noise=np.diag(np.ones(args.dimension)))
if args.random:
    pi.set_weights(np.random.rand((args.dimension, args.dimension)))
else:
    K = env.computeOptimalK()
    print("optimal K", K)
    pi.set_weights(K, action_noise)
policy_label = 'linear_gaussian_policy'
policy = lambda s: pi.act(s)
random_policy = lambda s: np.random.uniform(-args.max_rand_action, args.max_rand_action, size=2)
start_time = str(time.time())
n_samples_irl = [2, 5, 10, 20, 50, 100, 200, 500, 1000] #, 2000
out_dir = 'data/lqg_final/' + policy_label + "/" + start_time + "/"
demonstrations = np.max(n_samples_irl)
if args.dir_to_read != "":
    dir_to_read = args.dir_to_read
else:
    dir_to_read = out_dir
for exp in range(args.num_experiments):
    np.random.seed(exp)
    trajectories = []
    # if a input directory is specified, read the directory from there
    if args.dir_to_read == '':
        print("Collecting %d trajectories" % demonstrations)
        trajectories, save_keys, features_array = collect_trajectories(demonstrations, env, policy,
                                                                       args.diff_scales, args.random_init)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_dir + str(exp) + "_" + str(demonstrations) + '_trajectories.csv', 'w+',
                  newline='') as trajectories_file:
            dicti_writer = csv.DictWriter(trajectories_file, fieldnames=save_keys)
            dicti_writer.writeheader()
            for trajectory in trajectories:
                dicti_writer.writerows([sample for sample in trajectory])
    # read the expert demonstrations
    states, actions, next_states, _, features, dones = read_trajectories(
        dir_to_read + str(exp) + "_" + str(demonstrations) + '_trajectories.csv',
        all_columns=True,
        fill_size=args.horizon,
        cont_actions=True)
    # if we are employing gradient based irl compute the gradients
    if args.settings not in ['re_irl', 'csi']:
        if not args.load_gradients:
            grads, _ = compute_gradient(pi, states, actions, features,
                                        dones, args.horizon, args.gamma, features_idx, use_baseline=True,
                                        filter_gradients=True)
            if args.save_gradients:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)
                np.save(out_dir + '/estimated_gradients.npy', grads)
        else:
            grads = np.load(dir_to_read + '/estimated_gradients.npy')

        means = grads.mean(axis=0)
        _, s, v = np.linalg.svd(means)
        print("Mean Gradient:")
        print(means)
        print("Singular values")
        print(s)
    for num_samples in n_samples_irl:
        if args.settings not in ['re_irl', 'csi']:
            g = grads[0: num_samples, :, :]
        print("Considering %d samples." % num_samples)
        n = args.dimension
        for setting in settings:
            solver, params, result = setting_to_reqs[setting]
            if setting == 're_irl':
                ft = np.array(features)
                features_array = np.reshape(ft, [demonstrations, args.horizon, ft.shape[-1]])
                features_array = features_array[0: num_samples, :, :]
                # collect the random dataset
                _, _, features_random = collect_trajectories(num_samples,
                                                             env, random_policy, args.diff_scales)

                solver = RelativeEntropyIRL(gamma=args.gamma, horizon=args.horizon, reward_features=features_array,
                                            reward_random=features_random)
                weights = solver.fit(verbose=args.verbose)
                loss = -1
            elif setting == 'csi':
                sts = np.reshape(np.array(states), [demonstrations, args.horizon, len(states[0])])[:num_samples]
                acs = np.reshape(np.array(actions), [demonstrations, args.horizon, len(actions[0])])[:num_samples]
                ft = np.array(features)
                # discretize the actions before
                max_a = 10
                features_array = np.reshape(ft, [demonstrations, args.horizon, ft.shape[-1]])[:num_samples]
                discretized_actions = discretize_actions(acs, 2 * [max_a], 2 * [- max_a], max_a)
                mask = np.ones((num_samples, args.horizon))
                weights = csi(sts, discretized_actions, mask, features_array, args.gamma, use_heuristic=True)
                loss = -1
            else:
                weights, loss, _ = solver(g, **params)

            if args.verbose:
                print(setting + " Computed Weights with %d Episodes:" % demonstrations)
            Q = np.diag(weights[:n])
            R = np.diag(weights[n:])
            if np.isclose(R, 0).any():
                R = R + np.eye(n) * 0.01
            if np.isclose(Q, 0).any():
                Q = Q + np.eye(n) * 0.01
            env_test.set_costs(Q, R)
            try:
                k = env_test.computeOptimalK()
            except:
                print(Q)
                print(R)
                input()
            if args.verbose:
                print(setting + " Computed K with %d Episodes:" % demonstrations)

            J_sigma = env.computeJ(k, Sigma, args.eps_eval, random_init=False)
            J = env.computeJ(k, np.diag(np.zeros(args.dimension)), args.eps_eval, random_init=False)

            if args.verbose:
                print(setting + " Computed Performace with %d Episodes:" % demonstrations)
            diff_weights = np.linalg.norm(opt_weights - weights)

            diff_pi = np.linalg.norm(K - k)
            print("Finished " + str(num_samples) + "episodes with " + setting)
            result.append({
                "num_episodes": num_samples,
                "run": exp,
                "J_opt_sigma": opt_J_sigma,
                "J_opt": opt_J,
                "J_sigma": J_sigma,
                "J": J,
                "loss": loss,
                "weights": list(weights),
                "diff_weights": diff_weights,
                "diff_pi": diff_pi,
                "k": k.flatten()
            })

        for setting in settings:
            file_name = out_dir + setting + ".csv"
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            _, _, results = setting_to_reqs[setting]
            with open(file_name, 'w+', newline='') as out_file:
                dicti_writer = csv.DictWriter(out_file, fieldnames=kpis)
                dicti_writer.writeheader()
                dicti_writer.writerows([sample for sample in results])
