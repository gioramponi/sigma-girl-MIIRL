import os
import numpy as np
import argparse
import glob
from trajectories_reader import read_trajectories
from utils import compute_gradient, load_policy, estimate_distribution_params, filter_grads, algorithm_u
from algorithms.pgirl import make_loss_function, make_weights_assignment_function
from scipy.cluster.hierarchy import fcluster, dendrogram, linkage
import matplotlib.pyplot as plt
from scipy import optimize
from tqdm import tqdm

# SCRIPT TO RUN CLUSTERING GIVEN TRAJECTORIES OF SOME EXPERTS

# Directories where the agent policies, trajectories and gradients (if already calculated) are stored
# To add agents populate this dictionary and store the gradients in '/gradients/estimated_gradients.npy'
# Or if u want to calculate the gradients directly store the policy as a tf checkpoint in a file called best
# and the trajectories in the subfolder 'trajectories/<subfolder>/K_trajectories.csv'
agent_to_data = {
    "border": ["logs/cont_gridworld/trpo-rews-1.0_10.0_0.0/border/models/1565016041.9701474", []],
    "border3": ["logs/cont_gridworld/trpo-rews-1.0_10.0_0.0/border/models/1564755468.495175", []],
    "center": ["logs/cont_gridworld/trpo-rews-1.0_10.0_0.0/center/models/1565012444.4165735", []],
    "center3": ["logs/cont_gridworld/trpo-rews-1.0_10.0_0.0/center/models/1564758925.6475885", []],
    "up": ["logs/cont_gridworld/trpo-rews-1.0_10.0_0.0/up/models/1565016052.0781844", []],
    "up3": ["logs/cont_gridworld/trpo-rews-1.0_10.0_0.0/up/models/1564760818.2994974", []],
    "down": ["logs/cont_gridworld/trpo-rews-1.0_10.0_0.0/down/models/1565007021.7597847", []],
    "down3": ["logs/cont_gridworld/trpo-rews-1.0_10.0_0.0/down/models/1564758913.6936421", []],
    # 'border2': ["logs/cont_gridworld/gpomdp2/border", []],
    # 'center2': ["logs/cont_gridworld/gpomdp2/center", []],
    # 'up2': ["logs/cont_gridworld/gpomdp2/up", []],
    # 'down2': ["logs/cont_gridworld/gpomdp2/down", []],

    #'border2': ["logs/cont_gridworld/gpomdp3/border/dataset_20", []],
    # 'center2': ["logs/cont_gridworld/gpomdp3/center/dataset_20/", []],
    # 'up2': ["logs/cont_gridworld/gpomdp3/up/dataset_20", []],
    # 'down2': ["logs/cont_gridworld/gpomdp3/down/dataset_20", []],
}

agents = ["uniform", "tweet", "no_tweet"]
for i in range(10):
    for agent in agents:
        agent_to_data[agent + "_" + str(i)] = ["data/twitter/" + agent + "/1000_" + str(i), []]


def em_clustering(mus, sigmas, ids, lamb=1, num_clusters=2, num_objectives=3, max_iterations=100, tolerance=1e-3,
                  verbose=False, optimization_iterations=10, cluster_iterations=1):
    num_agents = len(mus)
    loss_functions = []
    tolerance = 1e-7
    for i, mu in enumerate(mus):
        loss_functions.append(make_loss_function(mu, sigmas[i], ids[i]))

    weight_calculator = make_weights_assignment_function(mus, sigmas, ids, num_objectives=num_objectives,
                                                         verbose=verbose,
                                                         num_iters=optimization_iterations)
    min_loss = np.inf
    best_assignment = None
    for c_it in range(cluster_iterations):
        p = np.random.uniform(size=(num_clusters, num_agents))
        p = p / np.sum(p, axis=0)
        it = 0
        omega = np.random.random((num_clusters, num_objectives))
        prev_assignment = None

        while it < max_iterations:
            it += 1
            # find best omega for assignment
            # equal to minimizing a function for each separate cluster
            for i in range(num_clusters):
                w, loss = weight_calculator(p[i])
                omega[i] = w

            print("Omegas at iteration %d: " % it)
            print(omega)

            # find best assignment for given Omega
            # best assignment is hard
            loss_value = 0
            for j, mu in enumerate(mus):
                p[:, j] = 0.
                minimum = np.inf
                # index = -1
                for i in range(num_clusters):
                    loss = loss_functions[j](omega[i])
                    if loss < minimum:
                        # index = i
                        minimum = loss
                    p[i, j] = -lamb * loss
                p[:, j] -= np.max(p[:, j])
                p[:, j] = np.exp(p[:, j])
                p[:, j] /= np.sum(p[:, j])
                loss_value += minimum
            # p2 = np.mean(p, axis=1)
            # print(p2)
            print("Assignment at iteration %d: " % it)
            print(p)
            print("Loss Value ", loss_value)
            if loss_value < min_loss:
                min_loss = loss_value
                best_assignment = np.copy(p)
            if prev_assignment is not None:
                diff = np.max(np.abs(prev_assignment - p))
                if diff < tolerance:
                    print("Converged at Iteration %d:" % it)
                    break
            # perturbation of assignment
            prev_assignment = np.copy(p)
            # p += np.random.normal(scale=0.1, size=p.shape)
            # p = np.clip(p, 0, 1)
            # p = p / np.sum(p, axis=0)
        if it == max_iterations:
            print("Finished %d iterations without converging" % max_iterations)
    best_weights = np.zeros(shape=(num_clusters, num_objectives))
    for i in range(num_clusters):
        w, loss = weight_calculator(best_assignment[i])
        best_weights[i] = w
    return best_assignment, best_weights, min_loss


def run_alternating_clustering(mus, sigmas, ids, num_clusters=2, num_objectives=3, max_iterations=100, tolerance=1e-3,
                               verbose=False, optimization_iterations=10, cluster_iterations=1):
    num_agents = len(mus)
    # tolerance = 1.e-10
    loss_functions = []
    for i, mu in enumerate(mus):
        loss_functions.append(make_loss_function(mu, sigmas[i], ids[i]))

    weight_calculator = make_weights_assignment_function(mus, sigmas, ids, num_objectives=num_objectives,
                                                         verbose=verbose,
                                                         num_iters=optimization_iterations)
    min_loss = np.inf
    best_assignment = None
    for c_it in range(cluster_iterations):
        # initial assignment
        if False:  # num_clusters == num_agents:
            p = np.eye(num_clusters)
        else:
            p = np.random.uniform(size=(num_clusters, num_agents))
            p = p / np.sum(p, axis=0)
        # Best Assignment
        omega = np.zeros(shape=(num_clusters, num_objectives))
        it = 0
        diff = np.inf
        prev_assignment = None

        while it < max_iterations:
            it += 1
            # find best omega for assignment
            # equal to minimizing a function for each separate cluster
            for i in range(num_clusters):
                w, loss = weight_calculator(p[i])
                omega[i] = w

            print("Omegas at iteration %d: " % it)
            print(omega)

            # find best assignment for given Omega
            # best assignment is hard
            loss_value = 0
            for j, mu in enumerate(mus):
                p[:, j] = 0.
                minimum = np.inf
                index = -1
                for i in range(num_clusters):
                    loss = loss_functions[j](omega[i])
                    if loss < minimum:
                        index = i
                        minimum = loss
                p[index, j] = 1
                loss_value += minimum
            print("Assignment at iteration %d: " % it)
            print(p)
            print("Loss Value ", loss_value)
            if loss_value < min_loss:
                min_loss = loss_value
                best_assignment = np.copy(p)
            if prev_assignment is not None:
                diff = np.max(np.abs(prev_assignment - p))
                if diff < tolerance:
                    print("Converged at Iteration %d:" % it)
                    break
            # perturbation of assignment
            prev_assignment = np.copy(p)
            # p += np.random.normal(scale=0.1, size=p.shape)
            # p = np.clip(p, 0, 1)
            # p = p / np.sum(p, axis=0)
        if it == max_iterations:
            print("Finished %d iterations without converging" % max_iterations)

    best_weights = np.zeros(shape=(num_clusters, num_objectives))
    for i in range(num_clusters):
        w, loss = weight_calculator(best_assignment[i])
        best_weights[i] = w
    return best_assignment, best_weights, min_loss


def solve_exact(mus, sigmas, ids, num_clusters=2, num_objectives=3, verbose=False, optimization_iterations=10):
    num_agents = len(mus)
    assert num_clusters <= num_agents
    weight_calculator = make_weights_assignment_function(mus, sigmas, ids, num_objectives=num_objectives,
                                                         verbose=verbose,
                                                         num_iters=optimization_iterations)
    min_loss = np.inf

    candidate_assignment = np.zeros((num_clusters, num_agents))
    candidate_omega = np.zeros((num_clusters, num_objectives))
    agents = np.array([i for i in range(num_agents)])
    for partition in algorithm_u(agents, num_clusters):
        candidate_loss = 0

        for i, p in enumerate(partition):
            candidate_assignment[i] = 0
            for index in p:
                candidate_assignment[i, index] = 1
            w, loss = weight_calculator(candidate_assignment[i])
            candidate_omega[i] = w
            candidate_loss += loss

        if candidate_loss < min_loss:
            min_loss = candidate_loss
            best_assignment = np.copy(candidate_assignment)
            best_omega = np.copy(candidate_omega)
    return best_assignment, best_omega, min_loss


def run_hierarchical_clustering(mus, sigmas, ids, num_clusters=2, num_objectives=3, verbose=False,
                                optimization_iterations=10, display=False, criterion='maxclust_monocrit'):
    num_agents = len(mus)

    weight_calculator = make_weights_assignment_function(mus, sigmas, ids, num_objectives=num_objectives,
                                                         verbose=verbose,
                                                         num_iters=optimization_iterations)
    # one hot encoding of agents
    X = np.eye(num_agents)

    def distance_func(i, j):
        assignment = np.logical_or(i, j)
        w, dist = weight_calculator(assignment)
        return dist

    Z = linkage(X,
                method='complete',  # dissimilarity metric: max distance across all pairs of
                # records between two clusters
                metric=distance_func
                )  # you can peek into the Z matrix to see how clusters are

    if display:
        # calculate full dendrogram and visualize it
        plt.figure(figsize=(30, 10))
        dendrogram(Z, labels=list(agent_to_data.keys()))
        plt.show()
    print(criterion)
    clusters = fcluster(Z, num_clusters, criterion='maxclust')

    # calculate weights and loss value
    best_assignment = np.zeros(shape=(num_clusters, num_agents))
    for i in range(num_agents):
        best_assignment[clusters[i] - 1, i] = 1.
    best_weights = np.zeros(shape=(num_clusters, num_objectives))
    loss_value = 0.
    for i in range(num_clusters):
        w, loss = weight_calculator(best_assignment[i])
        best_weights[i] = w
        loss_value += loss
    return best_assignment, best_weights, loss_value


def run_pso_clustering(mus, sigmas, ids, num_clusters=2, num_objectives=3, verbose=False, optimization_iterations=10,
                       num_particles=10):
    try:
        from psopy import minimize
    except ImportError:
        print("Psopy not installed: ")
        return 0, 0, 0
    num_agents = len(mus)

    weight_calculator = make_weights_assignment_function(mus, sigmas, ids, num_objectives=num_objectives,
                                                         verbose=verbose,
                                                         num_iters=optimization_iterations)

    # The objective function.
    def obj_func(assignment):
        assignment = assignment.reshape((num_clusters, num_agents))
        obj = 0
        for i in range(assignment.shape[0]):
            w, loss = weight_calculator(assignment[i])
            obj += loss
        return obj

    # The constraints.
    def simplex(p):
        p = p.reshape((num_clusters, num_agents))
        if np.isclose(np.sum(p, axis=0) - 1., 0.).all():
            return 0
        else:
            return -1

    cons = ({'type': 'ineq', 'fun': lambda p: np.min(1 - p)},
            {'type': 'ineq', 'fun': lambda p: np.min(p)},
            {'type': 'eq', 'fun': simplex})
    x0 = np.zeros((num_particles, num_clusters * num_agents))
    for i in range(num_particles):
        p = np.random.uniform(size=(num_clusters, num_agents))
        p = p / np.sum(p, axis=0)
        p = p.flatten()
        x0[i] = p
    res = minimize(obj_func, x0, constraints=cons, options={
        'g_rate': 1., 'l_rate': 1., 'max_velocity': 4., 'stable_iter': 50})
    print(res.x)
    best_assignment = res.x
    loss_value = 0.
    best_weights = np.zeros(shape=(num_clusters, num_objectives))
    best_assignment = best_assignment.reshape((num_clusters, num_agents))
    for i in range(num_clusters):
        w, loss = weight_calculator(best_assignment[i])
        best_weights[i] = w
        loss_value += loss
    return best_assignment, best_weights, loss_value


def run_soft_optimization(mus, sigmas, ids, num_clusters=2, num_objectives=3, verbose=False,
                          optimization_iterations=10, num_iters=10):
    num_agents = len(mus)

    weight_calculator = make_weights_assignment_function(mus, sigmas, ids, num_objectives=num_objectives,
                                                         verbose=False,
                                                         num_iters=optimization_iterations)

    # The objective function.
    def obj_func(assignment):
        # assignment = assignment.reshape((num_clusters, num_agents))
        obj = 0
        for i in range(num_clusters):
            w, loss = weight_calculator(assignment[i * num_agents: (i + 1) * num_agents])
            obj += loss
        return np.sqrt(obj)

    e = np.ones(num_clusters)
    e_2 = np.ones(num_agents)

    # The constraints.
    def simplex(p):
        p = p.reshape((num_clusters, num_agents))
        res = np.dot(e, p) - e_2
        return np.sum(res ** 2)

    cons = ({'type': 'eq', 'fun': simplex})
    bound = (0, 1)
    bounds = [bound] * num_clusters * num_agents
    evaluations = []
    i = 0
    pbar = tqdm(total=num_iters, desc="Assignment Optimization")
    while i < num_iters:
        p = np.random.uniform(size=(num_clusters, num_agents))
        p = p / np.sum(p, axis=0)
        x0 = p.flatten()
        # x0 = np.array([[1, 0, 1], [0, 1, 0]])
        res = optimize.minimize(obj_func,
                                x0,
                                method='SLSQP',
                                constraints=cons,
                                bounds=bounds,
                                options={'ftol': 1e-2, 'disp': verbose})
        if res.success:
            evaluations.append([res.x, obj_func(res.x)])
            pbar.update(1)
            i += 1
    evaluations = np.array(evaluations)
    min_index = np.argmin(evaluations[:, 1])
    x, y = evaluations[min_index, :]

    best_assignment = x
    loss_value = 0.
    best_weights = np.zeros(shape=(num_clusters, num_objectives))
    best_assignment = best_assignment.reshape((num_clusters, num_agents))
    for i in range(num_clusters):
        w, loss = weight_calculator(best_assignment[i])
        best_weights[i] = w
        loss_value += loss
    pbar.close()
    return best_assignment, best_weights, loss_value


def run_rbf_optimization(mus, sigmas, ids, num_clusters=2, num_objectives=3, verbose=False,
                         optimization_iterations=10, ):
    try:
        import rbfopt
        settings = rbfopt.RbfoptSettings(minlp_solver_path='/home/amarildo/Bonmin-stable/build/bin/bonmin',
                                         nlp_solver_path='/home/amarildo/Bonmin-stable/build/bin/ipopt')
        num_agents = len(mus)

        weight_calculator = make_weights_assignment_function(mus, sigmas, ids, num_objectives=num_objectives,
                                                             verbose=verbose,
                                                             num_iters=optimization_iterations)

        # The objective function.
        def obj_func(assignment):
            assignment = assignment.reshape((num_clusters, num_agents))
            obj = 0
            for i in range(assignment.shape[0]):
                w, loss = weight_calculator(assignment[i])
                obj += loss
            return obj

        num_opt_params = num_clusters * num_agents
        bb = rbfopt.RbfoptUserBlackBox(num_opt_params, np.array([0] * num_opt_params), np.array([1] * num_opt_params),
                                       np.array(['R'] * num_opt_params), obj_func)
        alg = rbfopt.RbfoptAlgorithm(settings, bb)
        val, x, itercount, evalcount, fast_evalcount = alg.optimize()

        print("Value:", val)
        print("Assignment:")
        print(x)
        print("Iteration Count:", itercount)
        print("Evaluation Count:", evalcount)
        print("Fast Evaluation Count:", fast_evalcount)
        loss_value = 0.
        best_weights = np.zeros(shape=(num_clusters, num_objectives))
        best_assignment = x.reshape((num_clusters, num_agents))
        for i in range(num_clusters):
            w, loss = weight_calculator(best_assignment[i])
            best_weights[i] = w
            loss_value += loss
        return best_assignment, best_weights, loss_value
    except ImportError:
        print("RBF optimizer not installed")
        return 0, 0, 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg_env = parser.add_argument_group('Environment')
    arg_env.add_argument('--horizon', type=int, default=50, help="Environment horizon")
    arg_env.add_argument('--gamma', type=float, default=0.9999, help="Discount factor")
    arg_env.add_argument('--continuous', action='store_true', help="Type of environment")
    arg_env.add_argument('--lqg', action='store_true', help='wether the environment is lqg')
    arg_env.add_argument('--ep_len', type=int, default=50, help='episodes lengths')

    arg_pol = parser.add_argument_group('Policy')
    arg_pol.add_argument("--init_logstd", type=float, default=-2, help="Policy variance")
    arg_pol.add_argument('--trainable_variance', action='store_true', help="Whether to train the policy variance")

    arg_irl = parser.add_argument_group('IRL')
    arg_irl.add_argument('--features_idx', default='', type=str, help='comma separated indexes of reward features'
                                                                      ' to consider')
    arg_irl.add_argument('--cov_estimation', action='store_true', help='Regularize covariance matrix')
    arg_irl.add_argument('--diag', action='store_true', help='Diagonal covariance matrix')
    arg_irl.add_argument('--identity', action='store_true', help='Identity covariance matrix')
    arg_irl.add_argument('--girl', action='store_true', help='use plain girl covariance model')
    arg_irl.add_argument('--mask', action='store_true', help='use mask in gradient computation')
    arg_irl.add_argument('--baseline', action='store_true', help='use baseline in gradient computation')
    arg_irl.add_argument('--filter_gradients', action='store_true', help='remove jacobian 0 jacobian rows')
    arg_irl.add_argument('--opt_iters', type=int, default=25, help='number of starting points in IRL optimization')

    arg_clusters = parser.add_argument_group('Cluster')
    arg_clusters.add_argument('--num_clusters', type=int, default=2, help='Number of clusters to divide agents')
    arg_clusters.add_argument('--tolerance', type=float, default=1e-3, help='Tolerance of clustering iterations')
    arg_clusters.add_argument('--max_iterations', type=int, default=100, help='Iterations of alternating clustering')
    arg_clusters.add_argument('--cluster_iterations', type=int, default=1,
                              help='Number of initializations of alternating clustering')
    arg_clusters.add_argument('--h_cluster', action='store_true', help='Perform hierarchical clustering')
    arg_clusters.add_argument('--pso', action='store_true', help='Perform particle swarm optimization clustering')
    arg_clusters.add_argument('--exact', action='store_true', help='Perform exact hard clustering')
    arg_clusters.add_argument('--em', action='store_true', help='Perform EM sort clustering')
    arg_clusters.add_argument('--slsqp', action='store_true', help='Perform SLSQP clustering')
    arg_clusters.add_argument('--rbf_opt', action='store_true', help='Perform rbf optimizer clustering')
    arg_clusters.add_argument('--display_cluster', action='store_true', help='Display hierarchical clusters')
    arg_clusters.add_argument('--clust_crit', type=str, choices=['maxclust', 'maxclust_monocrit'],
                              default='maxclust_monocrit', help='hierarchical clustering criteria')
    arg_clusters.add_argument('--num_particles', type=int, default=10, help='number of particles in pso algorithm')
    arg_clusters.add_argument('--tolerance', type=float, default=1e-3, help='stopping threshold in clustering')
    arg_clusters.add_argument('--s_iters', type=int, default=10, help='soft optimization iterations')

    arg_utils = parser.add_argument_group('Utils')
    arg_utils.add_argument('--verbose', action='store_true', help='log into the terminal')
    arg_utils.add_argument('--save_grad', action='store_true', help='save computed gradients')
    arg_utils.add_argument('--seed', type=int, default=1234)
    arg_utils.add_argument('--num_episodes', type=int, default=-1, help='number of episodes to consider')
    arg_utils.add_argument('--read_grads', action='store_true', help='read precomputed gradients instead of '
                                                                     'calculating from demonstrations')

    args = parser.parse_args()

    EPISODE_LENGTH = args.ep_len
    # np.random.seed(20)
    if args.features_idx == '':
        features_idx = [0, 1, 2]
    else:
        features_idx = [int(x) for x in args.features_idx.split(',')]
    num_objectives = len(features_idx)
    mus = []
    sigmas = []
    ids = []
    for i, agent in enumerate(agent_to_data.keys()):
        read_path = agent_to_data[agent][0]
        if not args.read_grads:
            paths = glob.glob(read_path + "/*/*trajectories.csv")
            for p in paths:
                states, actions, _, _, features, dones = \
                    read_trajectories(p, all_columns=True,
                                      fill_size=EPISODE_LENGTH,
                                      cont_actions=True)
                X_dim = len(states[0])
                model_path = read_path + "/best"
                linear = 'gpomdp' in model_path
                pi = load_policy(X_dim=X_dim, model=model_path, continuous=True, num_actions=2,
                                 trainable_variance=args.trainable_variance, init_logstd=args.init_logstd,
                                 linear=linear)
                if args.num_episodes > 0:
                    states = states[:EPISODE_LENGTH * args.num_episodes]
                    actions = actions[:EPISODE_LENGTH * args.num_episodes]
                    features = features[:EPISODE_LENGTH * args.num_episodes]
                    dones = dones[:EPISODE_LENGTH * args.num_episodes]
                estimated_gradients, _ = compute_gradient(pi, states, actions, features, dones,
                                                          EPISODE_LENGTH, args.gamma, features_idx,
                                                          verbose=args.verbose,
                                                          use_baseline=args.baseline,
                                                          use_mask=args.mask,
                                                          filter_gradients=args.filter_gradients,
                                                          normalize_f=False)
                if args.save_grad:
                    if not os.path.exists(read_path + '/gradients'):
                        os.makedirs(read_path + '/gradients')
                    np.save(read_path + '/gradients/estimated_gradients.npy', estimated_gradients)
                num_episodes, num_parameters, num_objectives = estimated_gradients.shape[:]
                mu, sigma = estimate_distribution_params(estimated_gradients=estimated_gradients,
                                                         diag=args.diag, identity=args.identity,
                                                         cov_estimation=args.cov_estimation,
                                                         other_options=[False, args.girl])
                id_matrix = np.identity(num_parameters)
                mus.append(mu)
                sigmas.append(sigma)
                ids.append(id_matrix)

        else:
            estimated_gradients = np.load(read_path + '/gradients/estimated_gradients.npy', allow_pickle=True)
            estimated_gradients = estimated_gradients[:, :, features_idx]
            if args.num_episodes > 0:
                estimated_gradients = estimated_gradients[:args.num_episodes, :, features_idx]
            if args.filter_gradients:
                estimated_gradients = filter_grads(estimated_gradients, verbose=args.verbose)
            num_episodes, num_parameters, num_objectives = estimated_gradients.shape[:]
            mu, sigma = estimate_distribution_params(estimated_gradients=estimated_gradients,
                                                     diag=args.diag, identity=args.identity,
                                                     cov_estimation=args.cov_estimation, other_options=[False, args.girl])
            id_matrix = np.identity(num_parameters)
            mus.append(mu)
            sigmas.append(sigma)
            ids.append(id_matrix)
    np.random.seed(args.seed)
    if args.h_cluster:
        P, Omega, loss = run_hierarchical_clustering(mus, sigmas, ids, num_clusters=args.num_clusters,
                                                     num_objectives=num_objectives, verbose=args.verbose,
                                                     optimization_iterations=args.opt_iters,
                                                     display=args.display_cluster, criterion=args.clust_crit)
    elif args.pso:
        P, Omega, loss = run_pso_clustering(mus, sigmas, ids, num_clusters=args.num_clusters,
                                            num_objectives=num_objectives, verbose=args.verbose,
                                            optimization_iterations=args.opt_iters,
                                            num_particles=args.num_particles)
    elif args.exact:
        agent_names = list(agent_to_data.keys())
        P, Omega, loss = solve_exact(mus, sigmas, ids, num_clusters=args.num_clusters,
                                     num_objectives=num_objectives, verbose=args.verbose,
                                     optimization_iterations=args.opt_iters, )
    elif args.slsqp:
        P, Omega, loss = run_soft_optimization(mus, sigmas, ids, num_clusters=args.num_clusters,
                                               num_objectives=num_objectives, verbose=args.verbose,
                                               optimization_iterations=args.opt_iters, num_iters=args.s_iters)
    elif args.rbf_opt:
        P, Omega, loss = run_rbf_optimization(mus, sigmas, ids, num_clusters=args.num_clusters,
                                              num_objectives=num_objectives, verbose=args.verbose,
                                              optimization_iterations=args.opt_iters)
    elif args.em:
        P, Omega, loss = em_clustering(mus, sigmas, ids, num_clusters=args.num_clusters,
                                       num_objectives=num_objectives, tolerance=args.tolerance,
                                       max_iterations=args.max_iterations, verbose=args.verbose,
                                       optimization_iterations=args.opt_iters,
                                       cluster_iterations=args.cluster_iterations)
    else:
        P, Omega, loss = run_alternating_clustering(mus, sigmas, ids, num_clusters=args.num_clusters,
                                                    num_objectives=num_objectives, tolerance=args.tolerance,
                                                    max_iterations=args.max_iterations, verbose=args.verbose,
                                                    optimization_iterations=args.opt_iters,
                                                    cluster_iterations=args.cluster_iterations)

    np.save("assignment.npy", P)
    np.save("weights.npy", Omega)
    agent_names = list(agent_to_data.keys())
    print("Agents")
    print(agent_names)
    print("Assignment:")
    print(P)
    print("Weights:")
    print(Omega)
    print("Loss:")
    print(loss)
