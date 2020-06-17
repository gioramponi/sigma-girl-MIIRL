import os
from algorithms.pgirl import solve_ra_PGIRL, solve_PGIRL, make_loss_function
from utils import compute_gradient, load_policy, filter_grads, estimate_distribution_params
import numpy as np
import re
import argparse
from trajectories_reader import read_trajectories


# SCRIPT TO RUN SINGLE IRL EXPERIMENTS USING ANY VERSION OF SIGMA-GIRL

NUM_ACTIONS = 4
GAMMA = 0.99
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


def check_weights(w):
    if np.isnan(w).any():
        return False
    return not np.isclose(w, 1, rtol=1e-2).any()


def set_global_seeds(seed):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(seed)
    np.random.seed(seed)


features = ['fast_area', 'slow_area', 'goal']
features_norm = [x + '_norm' for x in features]
features_norm_after = [x + "'" for x in features_norm]
weights_features = [x + '_w' for x in features]
weights_features_normalized = [x + '_normalized' for x in weights_features]
weights_features_normalized_again = [x + '_' for x in weights_features_normalized]


def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


def PGIRL(demonstrations=None, model=None, grad_path=None, features_idx=None, normalize_f=False, save_grad=True,
          opt_iters=10, compute_jacobian=False, estimate_weights=None, num_episodes=-1, pickled=False,
          continuous=False, num_hidden=8, num_layers=0, agent_name=None):
    if features_idx is None:
        features_idx = [0, 1, 2]

    logger = {}

    # Read or Calculate Gradient
    if args.read_grads:
        if grad_path != '':
            print("Reading gradients from:", grad_path)
            estimated_gradients = np.load(grad_path, allow_pickle=True)
        else:
            estimated_gradients = np.load(gradient_path + "estimated_gradients.npy", allow_pickle=True)
        estimated_gradients = estimated_gradients[:, :, features_idx]
        if num_episodes > 0:
            estimated_gradients = estimated_gradients[:num_episodes, :, :]
        if args.filter_gradients:
            estimated_gradients = filter_grads(estimated_gradients, verbose=args.verbose)
    else:
        if pickled:
            states_data = np.load(demonstrations + 'real_states.pkl', allow_pickle=True)
            actions_data = np.load(demonstrations + 'actions.pkl', allow_pickle=True)
            reward_data = np.load(demonstrations + 'rewards.pkl', allow_pickle=True)
            X_dataset = states_data[agent_name]
            y_dataset = actions_data[agent_name]
            r_dataset = reward_data[agent_name]
            print(np.sum(np.array(y_dataset)==1))
            input()

            dones_dataset = None
        else:
            # read trajectories
            X_dataset, y_dataset, _, _, r_dataset, dones_dataset = \
                read_trajectories(demonstrations, all_columns=True,
                                  fill_size=EPISODE_LENGTH,
                                  fix_goal=True,
                                  cont_actions=args.continuous or args.lqg)
        if num_episodes > 0:
            X_dataset = X_dataset[:EPISODE_LENGTH * num_episodes]
            y_dataset = y_dataset[:EPISODE_LENGTH * num_episodes]
            r_dataset = r_dataset[:EPISODE_LENGTH * num_episodes]

            if dones_dataset is not None:
                dones_dataset = dones_dataset[:EPISODE_LENGTH * num_episodes]

        X_dim = len(X_dataset[0])
        if continuous:
            y_dim = len(y_dataset[0])
        else:
            y_dim = 2
        # Create Policy
        linear = 'gpomdp' in model

        policy_train = load_policy(X_dim=X_dim, model=model, continuous=continuous, num_actions=y_dim, n_bases=X_dim,
                                   trainable_variance=args.trainable_variance, init_logstd=args.init_logstd,
                                   linear=linear, num_hidden=num_hidden, num_layers=num_layers)
        print('Loading dataset... done')
        # compute gradient estimation

        estimated_gradients, _ = compute_gradient(policy_train, X_dataset, y_dataset, r_dataset, dones_dataset,
                                                  EPISODE_LENGTH, GAMMA, features_idx,
                                                  verbose=args.verbose,
                                                  use_baseline=args.baseline,
                                                  use_mask=args.mask,
                                                  scale_features=args.scale_features,
                                                  filter_gradients=args.filter_gradients,
                                                  normalize_f=normalize_f)
    # ==================================================================================================================

    if save_grad:
        print("Saving gradients in ", gradient_path)
        np.save(gradient_path + 'estimated_gradients.npy', estimated_gradients)

    # solve PGIRL or Rank Approx PGIRL
    if args.girl:
        weights_girl, loss_girl = solve_PGIRL(estimated_gradients, verbose=args.verbose)
        estimate_weights = weights_girl
    if args.rank_approx:
        weights, loss, jacobian = solve_ra_PGIRL(estimated_gradients, verbose=args.verbose,
                                                 cov_estimation=args.cov_estimation, diag=args.diag,
                                                 identity=args.identity, num_iters=opt_iters,
                                                 compute_jacobian=compute_jacobian,
                                                 other_options=[False, False, args.masked_cov]
                                                 )
        if estimate_weights is not None or args.girl:
            mu, sigma = estimate_distribution_params(estimated_gradients=estimated_gradients,
                                                     diag=args.diag, identity=args.identity,
                                                     cov_estimation=args.cov_estimation,
                                                     girl=False, other_options=[False, False, args.masked_cov])

            id_matrix = np.identity(estimated_gradients.shape[1])
            lf = make_loss_function(mu, sigma, id_matrix)
            estimated_loss = lf(estimate_weights)

        if compute_jacobian:
            print("Jacobian Rank:")
            print(np.linalg.matrix_rank(jacobian))
            print("Jacobian s:")
            _, s, _ = np.linalg.svd(jacobian)
            print(s)

    else:
        weights, loss = solve_PGIRL(estimated_gradients, verbose=args.verbose)

    print("Weights:", weights)
    print("Loss:", loss)
    if args.girl:
        print("Weights Girl:", weights_girl)
        print("Loss Girl:", loss_girl)
    if estimate_weights is not None or args.girl:
        print("Loss in weights given:", estimated_loss)
    return logger, weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=0, help='number of hidden layers of mlp')
    parser.add_argument('--num_hidden', type=int, default=8, help='number of hidden units per layer')
    parser.add_argument('--agent_name', type=str, default='', help='name of the agent')
    parser.add_argument('--out_dir', type=str, default='default', help='output_dir')
    parser.add_argument('--demonstrations', type=str, default='logs/', help='where to read demonstrations')
    parser.add_argument('--model_name', type=str, default='models/trpo/cont/gridworld/checkpoint_240',
                        help='path to policy to load')
    parser.add_argument('--features_idx', default='', type=str, help='commma separated indexes of the reward features'
                                                                     ' to consider, default: consider all')
    parser.add_argument('--estimate_weights', default='', type=str, help='estimate GIRL loss at the found weights')
    parser.add_argument('--debug', action='store_true', help='display debug info of the policy model')
    parser.add_argument('--verbose', action='store_true', help='log information in terminal')
    parser.add_argument('--ep_len', type=int, default=20, help='episode lengths')
    parser.add_argument('--num_episodes', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--beta', type=float, default=1.0, help='inverse of temperature of Boltzman policy')
    parser.add_argument('--opt_iters', type=int, default=25, help='number of optimization iterations')
    parser.add_argument('--save_grad', action='store_true', help='save the computed gradients')
    parser.add_argument('--pos_weights', action='store_true', help='constrain to positive weights')
    parser.add_argument('--compute_jacobian', action='store_true', help='compute the jacobian at the sigma=girl optimum')
    parser.add_argument('--mask', action='store_true', help='mask trajectories for baseline computation')
    parser.add_argument('--baseline', action='store_true', help='use a baseline for GPOMDP gradient computation')
    parser.add_argument('--scale_features', type=int, default=1)
    parser.add_argument('--filter_gradients', action='store_true', help='remove 0 rows of the jacobian')
    parser.add_argument('--continuous', action='store_true', help='the action space is continuous')
    parser.add_argument('--lqg', action='store_true', help='the demonstrations are from lqg env')
    parser.add_argument('--trainable_variance', action='store_true', help='fit the noise of the policy')
    parser.add_argument("--init_logstd", type=float, default=-1, help='initial noise of the model')
    parser.add_argument('--rank_approx', action='store_true', help='use sigma girl')
    parser.add_argument('--cov_estimation', action='store_true', help='Regularize the sample covariance matrix')
    parser.add_argument('--masked_cov', action='store_true', help='use block covariance model')
    parser.add_argument('--diag', action='store_true', help='use diagonal covariance model')
    parser.add_argument('--girl', action='store_true', help='use plain girl covariance model')
    parser.add_argument('--identity', action='store_true', help='use identity covariance model')
    parser.add_argument('--read_grads', action='store_true', help='read the precomputed gradients, avoiding gradient'
                                                                  ' computation')
    parser.add_argument('--pickled', action='store_true', help='wether the demonstration data are pickled or csv format')
    parser.add_argument('--grad_path', default='', type=str, help='path of gradients to read')
    args = parser.parse_args()

    EPISODE_LENGTH = args.ep_len

    if args.estimate_weights == '':
        estimate_weights = None
    else:
        estimate_weights = [float(x) for x in args.estimate_weights.split(',')]

    if args.features_idx == '':
        features_idx = None
    else:
        features_idx = [float(x) for x in args.features_idx.split(',')]
    out_dir = "/".join(args.model_name.split('/')[:-1])
    gradient_path = out_dir + "/gradients/"
    if args.save_grad:

        if not os.path.exists(gradient_path):
            os.makedirs(gradient_path)
    set_global_seeds(args.seed)

    log, weights = PGIRL(demonstrations=args.demonstrations, model=args.model_name, grad_path=args.grad_path,
                         features_idx=features_idx, save_grad=args.save_grad, opt_iters=args.opt_iters,
                         compute_jacobian=args.compute_jacobian, estimate_weights=estimate_weights,
                         num_episodes=args.num_episodes, pickled=args.pickled, continuous=args.continuous,
                         num_hidden=args.num_hidden, num_layers=args.num_layers,agent_name=args.agent_name)
    print("Weights:", weights)
    np.save(out_dir + '/weights.npy', weights)
