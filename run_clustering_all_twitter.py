import numpy as np
from utils import compute_gradient, load_policy, estimate_distribution_params
from run_clustering import em_clustering
import argparse
import pickle

# Directories where the agent policies, trajectories and gradients (if already calcualted) are stored
# To add agents populate this dictionary and store the gradients in '/gradients/estimated_gradients.npy'
# Or if u want to calculate the gradients directly store the policy as a tf checkpoint in a file called best
# and the trajectories in the subfolder 'trajectories/<subfolder>/K_trajectories.csv'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')
    parser.add_argument('--num_hidden', type=int, default=8, help='number of hidden units')
    parser.add_argument('--n_experiments', type=int, default=1, help='number of experiments')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--verbose', action='store_true', help='print logs in console')
    parser.add_argument('--ep_len', type=int, default=20, help='episode length')
    parser.add_argument('--save_grad', action='store_true', help='save computed gradients')
    parser.add_argument('--mask', action='store_true', help='mask timesteps for baseline in gradient computation')
    parser.add_argument('--baseline', action='store_true', help='use baseline in gradient computation')
    parser.add_argument('--scale_features', type=int, default=1, help='rescale features in gradient computation')
    parser.add_argument('--filter_gradients', action='store_true', help='regularize jacobian matrix')
    parser.add_argument('--trainable_variance', action='store_true', help='fit the variance of the policy')
    parser.add_argument("--init_logstd", type=float, default=-1, help='initial policy variance')
    parser.add_argument('--save_path', type=str, default='data/twitter', help='path to save the model')
    args = parser.parse_args()

    n_experiments = args.n_experiments
    results = []
    n_agents = 1
    # where the demonstrations are
    demonstrations = 'data_twitter/'
    agent_to_data = ['pcastr', 'kotti_sasikanth', '_LXAI', 'sebganaya', 'braincor', 'britbot_art', 'suraj_kja',
                     'negar_rz', 'Aditya_elite', 'sara_hlt', 'jigarkdoshi', 'dh7net', 'estebanuribe',
                     'KlimZaporojets', 'mohitban47', 'deeplearningldn']
    num_objectives = 3
    states_data = np.load(demonstrations + 'real_states4.pkl', allow_pickle=True)
    actions_data = np.load(demonstrations + 'actions2.pkl', allow_pickle=True)
    reward_data = np.load(demonstrations + 'rewards4.pkl', allow_pickle=True)
    features_idx = [0, 1, 2]
    GAMMA = args.gamma
    for exp in range(n_experiments):
        P_true = np.array([[1, 1, 0, 0, 0] * n_agents, [0, 0, 1, 1, 1] * n_agents], dtype=np.float)
        print("Experiment %s" % (exp+1))
        estimated_gradients_all = []
        for agent_name in agent_to_data:
            X_dataset = states_data[agent_name]
            y_dataset = actions_data[agent_name]
            r_dataset = reward_data[agent_name]
            X_dim = len(X_dataset[0])
            y_dim = 2
            # Create Policy
            model = 'bc/models/' + agent_name + '/2000_22/best'
            linear = 'gpomdp' in model
            print('load policy..')
            policy_train = load_policy(X_dim=X_dim, model=model, continuous=False, num_actions=y_dim,
                                       n_bases=X_dim,
                                       trainable_variance=args.trainable_variance, init_logstd=args.init_logstd,
                                       linear=linear, num_hidden=args.num_hidden, num_layers=args.num_layers)
            print('Loading dataset... done')
            # compute gradient estimation
            estimated_gradients, _ = compute_gradient(policy_train, X_dataset, y_dataset, r_dataset, None,
                                                      args.ep_len, GAMMA, features_idx,
                                                      verbose=args.verbose,
                                                      use_baseline=args.baseline,
                                                      use_mask=args.mask,
                                                      scale_features=args.scale_features,
                                                      filter_gradients=args.filter_gradients,
                                                      normalize_f=False)
            estimated_gradients_all.append(estimated_gradients)
        # ==================================================================================================================

            if args.save_grad:
                print("Saving gradients in ", args.save_path)
                np.save(args.save_path + '/estimated_gradients.npy', estimated_gradients)
        mus = []
        sigmas = []
        ids = []

        for i, agent in enumerate(agent_to_data):
            num_episodes, num_parameters, num_objectives = estimated_gradients_all[i].shape[:]
            mu, sigma = estimate_distribution_params(estimated_gradients=estimated_gradients_all[i],
                                                    diag=False, identity=False, other_options=[False, False],
                                                    cov_estimation=True)
            id_matrix = np.identity(num_parameters)
            mus.append(mu)
            sigmas.append(sigma)
            ids.append(id_matrix)

        P, Omega, loss = em_clustering(mus, sigmas, ids, num_clusters=3,
                                       num_objectives=num_objectives,
                                       optimization_iterations=1)
        print(P)
        results.append((P, Omega, loss))
    with open(args.save_path + '/results.pkl', 'wb') as handle:
        pickle.dump(results, handle)
