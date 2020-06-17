import numpy as np
import os
from time import sleep
from envs.continuous_gridword import GridWorldAction
from estimators.gradient_descent import Adam
from algorithms.pgirl import solve_ra_PGIRL
import argparse
from joblib import Parallel, delayed
from algorithms.REIRL import RelativeEntropyIRL
from algorithms.CSI import csi
import pickle
from plot_gridworld import plot_grid

# SCRIPT TO COLLECT TRAJECTORIES, TRAIN POLICIES AND RUN IRL ALGORITHMS IN THE GRIDWORLD ENVIRONMENT

def create_batch_trajectories(env, batch_size, len_trajectories, param, variance, render=False):
    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.shape[0]

    states = np.zeros((batch_size, len_trajectories, state_dim))
    actions = np.zeros((batch_size, len_trajectories, action_dim))
    rewards = np.zeros((batch_size, len_trajectories))
    mask = np.ones((batch_size, len_trajectories))
    reward_features = np.zeros((batch_size, len_trajectories, 3))

    for batch in range(batch_size):
        state = env.reset(rbf=True)

        if render and batch == 0:
            env._render()
            sleep(0.1)

        for t in range(len_trajectories):
            action = np.random.multivariate_normal(np.dot(param.T, state), np.eye(action_dim) * variance)
            next_state, reward, done, info = env.step(action, rbf=True)

            if render and batch == 0:  # render:
                env._render()
                sleep(0.1)
            states[batch, t] = state
            actions[batch, t] = action
            rewards[batch, t] = reward
            reward_features[batch, t] = info['features']

            if done:
                states[batch, t + 1:] = state
                actions[batch, t + 1:] = action
                reward_features[batch, t + 1:] = info['features'][:3]
                mask[batch, t + 1:] = 0.
                break

            state = next_state

    return states, actions, rewards, reward_features, mask


def gradient_est(param, batch_size, len_trajectories, states, actions, var_policy):
    gradients = np.zeros((batch_size, len_trajectories, param.shape[0], param.shape[1]))

    for b in range(batch_size):
        for t in range(len_trajectories):
            action = actions[b, t]
            state = states[b, t]
            mean = np.dot(param.T, state)
            gradients[b, t, :, :] = np.outer(action - mean, state).T / var_policy
    return gradients


def policy_gradient_est(param, batch_size, len_trajectories, states, actions, rewards, reward_features, mask, gamma,
                        return_mean=True, return_ravel=False):
    if reward_features is not None:
        rewards = reward_features
    else:
        rewards = rewards[:, :, None]

    discount_factor_timestep = gamma ** np.arange(len_trajectories)
    discounted_reward = discount_factor_timestep[None, :, None] * rewards * mask[:, :, None]  # (N, T, k)

    gradients = gradient_est(param, batch_size, len_trajectories, states, actions, var_policy)  # (N, T, q, d)
    gradient_timestep = np.cumsum(gradients * mask[:, :, None, None], axis=1)  # (N, T, q, d)

    baseline_den = np.mean(gradient_timestep ** 2, axis=0)[:, :, :, None] + 1e-24  # (T, q, d, 1)
    baseline_num = np.mean(gradient_timestep[:, :, :, :, None] ** 2 * discounted_reward[:, :, None, None, :],
                           axis=0)  # (T, q, d, k)
    baseline = baseline_num / baseline_den

    gradient_per_episode = np.sum(gradient_timestep[:, :, :, :, None] * (discounted_reward[:, :, None, None, :]
                                                                         - baseline[None, :, :, :, :]),
                                  axis=1)  # (q, d, k)

    if return_mean:
        gradient = np.mean(gradient_per_episode, axis=0)
        if return_ravel:
            gradient = gradient.reshape(-1, gradient.shape[-1])
    else:
        gradient = gradient_per_episode
        if return_ravel:
            gradient = gradient.reshape(gradient.shape[0], -1, gradient.shape[-1])

    return gradient.squeeze()


def gpomdp(env, num_batch, batch_size, len_trajectories, initial_param, gamma, var_policy, verbose=False, render=False):
    param = np.array(initial_param)
    results = np.zeros(num_batch)

    discount_factor_timestep = gamma ** np.arange(len_trajectories)

    gradient = np.zeros_like(param)
    rewards__ = np.zeros(num_batch)
    gradients__ = np.zeros(num_batch)
    lens__ = np.zeros(num_batch)

    optimizer = Adam(learning_rate=0.05, ascent=True)
    optimizer.initialize(param)

    for i in range(num_batch):
        if i > 0:
            param = optimizer.update(gradient)
            # param += 0.05*gradient
        states, actions, rewards, _, mask = create_batch_trajectories(env, batch_size, len_trajectories, param,
                                                                      var_policy, render=render)

        gradient = policy_gradient_est(param, batch_size, len_trajectories, states, actions, rewards, None, mask, gamma)

        results[i] = np.mean(np.sum(discount_factor_timestep[None, :] * rewards * mask, axis=1), axis=0)

        if verbose:
            gradients__[i] = np.linalg.norm(gradient.ravel())
            rewards__[i] = np.mean(np.sum(rewards, axis=1))
            lens__[i] = np.mean(np.sum(mask, axis=1))
            print('Ite %s - Grad %s - Rewards %s - Lens %s' % (i, gradients__[i], rewards__[i], lens__[i]))

    return param, results, states, rewards__, gradients__


def compute_feature_expectations(len_trajectories, reward_features, mask, gamma):
    discount_factor_timestep = gamma ** np.arange(len_trajectories)
    return np.sum(discount_factor_timestep[None, :, None] * mask[:, :, None] * reward_features, axis=1)


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


def _perform_irl(i, N_traj_irl, n_samples_irl, horizon, fail_prob, save_gradients, param, rew_weights, var_policy,
                 save_path='.', load_gradients=False, load_path='.', agents_to_params=None, num_batches=100,
                 batch_size=50, eval_episodes=100,):
    if agents_to_params is None:
        agents_to_params = {
            'pgirl': (solve_ra_PGIRL, {'girl': True, 'other_options': [False, True, False]}, []),
            'ra_pgirl_diag': (solve_ra_PGIRL, {"diag": True}, []),
            # 'ra_pgirl_full': (solve_ra_PGIRL, {'seed': args.seed}, []),
            'ra_pgirl_cov_estimation': (solve_ra_PGIRL, {"cov_estimation": True}, []),
            're_irl': (RelativeEntropyIRL, {}, []),
            'csi': (csi, {}, []),
            'ra_pgirl_identity': (solve_ra_PGIRL, {"identity": True}, []),
        }

    agents = sorted(agents_to_params.keys())

    num_agents = len(agents)
    np.random.seed(i)
    with open(save_path + '/agents.pkl', 'wb') as handle:
        pickle.dump(agents, handle)

    res = np.zeros((len(n_samples_irl), num_agents))
    weights_diff = np.zeros((len(n_samples_irl),  num_agents))

    env = GridWorldAction(shape=shape, rew_weights=rew_weights,
                           randomized_initial=True, horizon=horizon,
                           n_bases=n_basis, fail_prob=fail_prob,
                           border_width=1)
    state_dim = np.prod(env.observation_space.shape)

    print("\nIRL Experiment %s" % i)
    print("Collecting Trajectories and Computing Gradients...")

    env_irl = GridWorldAction(shape=shape, rew_weights=[0., 0., 0.],
                              randomized_initial=True, horizon=horizon,
                              n_bases=n_basis, fail_prob=fail_prob,
                              border_width=1)

    if load_gradients:
        estimated_gradients = np.load(load_path + "/gridworld_grads_%s_%s_%s_%s.npy" % tuple(rew_weights + [i]))
        if 'csi' in agents:
            states, actions, _, reward_features, mask = create_batch_trajectories(env_irl, batch_size=N_traj_irl,
                                                                                      len_trajectories=horizon,
                                                                                      param=param,
                                                                                      variance=var_policy)

            discretized_actions = discretize_actions(actions, env.action_high, env.action_low, n_bins_per_dim=3)
    else:
        states, actions, _, reward_features, mask = create_batch_trajectories(env_irl, batch_size=N_traj_irl,
                                                                                  len_trajectories=horizon,
                                                                                  param=param,
                                                                                  variance=var_policy)

        discretized_actions = discretize_actions(actions, env.action_high, env.action_low, n_bins_per_dim=3)

        estimated_gradients = policy_gradient_est(param, batch_size=N_traj_irl, len_trajectories=horizon,
                                                   states=states, actions=actions, rewards=None,
                                                   reward_features=reward_features, mask=mask, gamma=gamma,
                                                   return_mean=False, return_ravel=True)

        # feature_exp1 = compute_feature_expectations(horizon, reward_features1, mask1, gamma)

        if save_gradients:
            np.save(save_path + "/gridworld_grads_%s_%s_%s_%s.npy" % tuple(rew_weights + [i]), estimated_gradients)

    print("Collected trajectories")
    print("States:", states.shape)
    # Collect trajectories for REIRL
    _, _, _, reward_features_random, mask = create_batch_trajectories(env_irl, batch_size=100,
                                                                      len_trajectories=horizon,
                                                                      param=np.zeros_like(param),
                                                                      variance=1)

    for j, n in enumerate(n_samples_irl):
        print("\n\tIRL Episodes: %s" % n)

        all_weights = []
        for k, agent in enumerate(agents):
            solver, params, result = agents_to_params[agent]
            print("Solving ", agent)
            if agent == 'csi':
                sts = np.reshape(np.array(states), [N_traj_irl, horizon, state_dim])[:n]
                ft = np.array(reward_features)
                # discretize the actions before
                features_array = np.reshape(ft, [N_traj_irl, horizon, ft.shape[-1]])[:n]
                disc_actions = discretized_actions[:n]
                mask = np.ones((n, horizon))
                weights = csi(sts, disc_actions, mask, features_array, args.gamma, use_heuristic=True)
            elif agent == 're_irl':
                ft = np.array(reward_features)
                features_array = np.reshape(ft, [N_traj_irl, args.horizon, ft.shape[-1]])
                features_array = features_array[0: n, :, :]
                # collect the random dataset
                features_random = reward_features_random[:n]

                solver = RelativeEntropyIRL(gamma=args.gamma, horizon=args.horizon, reward_features=features_array,
                                            reward_random=features_random)
                weights = solver.fit(verbose=False, gradient=False)

            else:
                g = estimated_gradients[:n]
                # weights, loss, _ = solver(g, seed=i, **params)
                try:
                    weights, loss, _ = solver(g, seed=i, **params)
                except:
                    weights = np.ones_like(rew_weights) / len(rew_weights)
                    loss = -1
            weights_diff[j, k] = np.linalg.norm(weights - rew_weights)
            all_weights.append(weights)
        if train_after_irl:
            for k, (name, weights) in enumerate(zip(agents, all_weights)):
                print("\t\tTraining Agent with %s weights..." % name)

                env_train_ = GridWorldAction(shape=shape, rew_weights=weights,
                                             randomized_initial=True, horizon=horizon,
                                             n_bases=n_basis, fail_prob=fail_prob,
                                             border_width=1)

                params_, _, _, _, _ = gpomdp(env_train_, num_batch=num_batches, batch_size=batch_size,
                                             len_trajectories=horizon,
                                             initial_param=np.zeros((np.prod(n_basis), 2)),
                                             gamma=gamma, var_policy=var_policy, verbose=False)

                _, results_, _, _, _ = gpomdp(env, num_batch=1, batch_size=50, len_trajectories=horizon,
                                              initial_param=params_,
                                              gamma=gamma, var_policy=var_policy, verbose=False)

                print("\t\tPerformance %s: %s" % (name, results_[0]))

                res[j, k] = results_[0]

    return res, weights_diff


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=int, default=100, help='length of the episodes')
    parser.add_argument('--gamma', type=float, default=0.999, help='discount factor')
    parser.add_argument('--var_policy', type=float, default=0.1, help='variance of the policy')
    parser.add_argument('--shape', type=int, nargs='+', default=[], help='shape of gird')
    parser.add_argument('--n_basis', type=int, nargs='+', default=[], help='number of rbf basis for the state '
                                                                           'representation')
    parser.add_argument('--fail_prob', type=float, default=0.1, help='stochasticity of the environment')
    parser.add_argument('--load_policy', action='store_true', help='load a pretrained policy')
    parser.add_argument('--load_path', type=str, default='data/gridworld', help='path to model to load')
    parser.add_argument('--num_batches', type=float, default=30, help='Number of interations of forward rl training')
    parser.add_argument('--batch_size', type=float, default=100, help='Number of episodes per batch')
    parser.add_argument('--save_policy', action='store_true', help='save the trained policy')
    parser.add_argument('--save_path', type=str, default='data/gridworld2', help='path to save the model')
    parser.add_argument('--save_gradients', action='store_true', help='save the computed gradients')
    parser.add_argument('--load_gradients', action='store_true', help='load the precomputed gradients')
    parser.add_argument('--train_after_irl', action='store_true', help='train with the computed rewards')
    parser.add_argument('--n_experiments', type=int, default=2, help='number of experiments to perform')
    parser.add_argument('--eval_episodes', type=int, default=50, help='number of evaluation episodes')
    parser.add_argument('--n_jobs', type=int, default=1, help='number of parallel jobs')
    parser.add_argument('--render_policy', action='store_true', help='render the interaction with the environment')
    parser.add_argument('--settings', type=str, default='', help='comma separated values of the settings to execute')
    parser.add_argument('--plot_results', action='store_true', help='plot the results')

    args = parser.parse_args()

    if len(args.shape) == 0 or len(args.shape) > 2:
        shape = [5, 5]
    elif len(args.shape) == 1:
        shape = [args.shape, args.shape]
    else:
        shape = args.shape

    if len(args.n_basis) == 0 or len(args.n_basis) > 2:
        n_basis = np.array([shape[0], 2 * shape[1]])
    elif len(args.n_basis) == 1:
        n_basis = [args.n_basis, 2 * args.n_basis]
    else:
        n_basis = args.n_basis

    n_samples_irl = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    N_traj_irl = max(n_samples_irl)

    if args.settings != '':
        settings = [x for x in args.settings.split(',')]
    n_experiments = args.n_experiments
    horizon = args.horizon
    gamma = args.gamma
    var_policy = args.var_policy
    fail_prob = args.fail_prob

    load_policy = args.load_policy
    save_policy = args.save_policy
    save_gradients = args.save_gradients
    train_after_irl = args.train_after_irl

    rew_weights = np.array([1., 100., 0.])

    rew_weights /= np.sum(rew_weights)

    rew_weights = rew_weights.tolist()

    agents_to_params = {
        'pgirl': (solve_ra_PGIRL, {'girl': True, 'other_options': [False, True, False]}, []),
        'ra_pgirl_diag': (solve_ra_PGIRL, {"diag": True}, []),
        # 'ra_pgirl_full': (solve_ra_PGIRL, {}, []),
        'ra_pgirl_cov_estimation': (solve_ra_PGIRL, {"cov_estimation": True}, []),
        #'ra_pgirl_convex': (solve_ra_PGIRL, {'other_options': [False, False, False, True]}, []),
        're_irl': (RelativeEntropyIRL, {}, []),
        'csi': (csi, {}, []),
        #'ra_pgirl_identity': (solve_ra_PGIRL, {"identity": True}, []),
    }
    agents = sorted(agents_to_params.keys())
    num_agents = len(agents)
    res = np.zeros((n_experiments, len(n_samples_irl), num_agents))
    all_weights = np.zeros((n_experiments, len(n_samples_irl), num_agents))

    env = GridWorldAction(shape=shape, rew_weights=rew_weights,
                          randomized_initial=True, horizon=horizon,
                          n_bases=n_basis, fail_prob=fail_prob,
                          border_width=1)
    if load_policy:
        print("Loading policy...")
        param = np.load(args.load_path + "/gridworld_param_%s_%s_%s_fail=%s.npy" % tuple(rew_weights + [fail_prob]))
    else:
        # Train expert

        param, _, _, _, _ = gpomdp(env, num_batch=args.num_batches, batch_size=args.batch_size,
                                    len_trajectories=horizon,
                                    initial_param=np.zeros((np.prod(n_basis), 2)),
                                    gamma=gamma, var_policy=var_policy, verbose=True)

        if save_policy:
            np.save(args.save_path + "/gridworld_param_%s_%s_%s_fail=%s.npy" % tuple(rew_weights + [fail_prob]),
                    param)


    if args.render_policy:
        _, _, _, _, _ = gpomdp(env, num_batch=100, batch_size=1, len_trajectories=horizon,
                               initial_param=param,
                               gamma=gamma, var_policy=0., verbose=False, render=True)

    print("Evaluating Policies")
    _, results, _, _, _ = gpomdp(env, num_batch=1, batch_size=args.eval_episodes, len_trajectories=horizon,
                                  initial_param=param,
                                  gamma=gamma, var_policy=var_policy, verbose=False, render=False)

    print(results)

    all_args = N_traj_irl, n_samples_irl, horizon, fail_prob, save_gradients, param, rew_weights, var_policy, \
               args.save_path, args.load_gradients, args.load_path, agents_to_params, args.num_batches, \
               args.batch_size, args.eval_episodes

    all_results = Parallel(n_jobs=args.n_jobs, verbose=51)(
        delayed(_perform_irl)(i, *all_args) for i in range(n_experiments))
    # all_results = [_perform_irl(0, *all_args)]
    for i in range(n_experiments):
        res[i], all_weights[i] = all_results[i]

    print(res, all_weights)

    np.save(args.save_path + '/gridworld_res_irl_all_%s.npy' % fail_prob, res)
    np.save(args.save_path + '/gridworld_res_irl_w_all_%s.npy' % fail_prob, all_weights)

    if args.plot_results:
        plot_grid(n_samples_irl, res, all_weights, agents, save_path=args.save_path)
