import numpy as np
from qpsolvers import solve_qp
from scipy import linalg
from scipy import optimize
import scipy
from utils import estimate_cov, estimate_distribution_params
from tqdm import tqdm
import scipy.linalg as scila


def solve_PGIRL(estimated_gradients, verbose=False, solver='quadprog', seed=1234,):
    num_episodes, num_parameters, num_objectives = estimated_gradients.shape[:]
    mean_gradients = np.mean(estimated_gradients, axis=0)
    ns = scipy.linalg.null_space(mean_gradients)
    P = np.dot(mean_gradients.T, mean_gradients)
    if ns.shape[1] > 0:
        if (ns >= 0).all() or (ns <= 0).all():

            print("Jacobian has a null space:", ns[:, 0] / np.sum(ns[:, 0]))
            weights = ns[:, 0] / np.sum(ns[:, 0])
            loss = np.dot(np.dot(weights.T, P), weights)
            return weights, loss
        else:
            weights = solve_polyhedra(ns)
            print("Null space:", ns)
            if weights is not None and (weights!=0).any():
                print("Linear programming sol:", weights)
                weights = np.dot(ns, weights.T)
                weights = weights / np.sum(weights)
                loss = np.dot(np.dot(weights.T, P), weights)
                print("Weights from non positive null space:", weights)
                return weights, loss
            else:
                print("Linear prog did not find positive weights")

    q = np.zeros(num_objectives)
    A = np.ones((num_objectives, num_objectives))
    b = np.ones(num_objectives)
    G = np.diag(np.diag(A))
    h = np.zeros(num_objectives)
    normalized_P = P / np.linalg.norm(P)
    try:
        weights = solve_qp(P, q, -G, h, A=A, b=b, solver=solver)
    except ValueError:
        try:
            weights = solve_qp(normalized_P, q, -G, h, A=A, b=b, solver=solver)
        except:
            #normalize matrix

            print("Error in Girl")
            print(P)
            print(normalized_P)
            u, s, v = np.linalg.svd(P)
            print("Singular Values:", s)
            ns = scipy.linalg.null_space(mean_gradients)
            print("Null space:", ns)

            weights, loss = solve_girl_approx(P, seed=seed)
    loss = np.dot(np.dot(weights.T, P), weights)
    if verbose:
        print('loss:', loss)
        print(weights)
    return weights, loss


def solve_ra_PGIRL(estimated_gradients, cov_estimation=False, diag=False, identity=False, seed=None, verbose=False,
                   num_iters=10, compute_jacobian=False, girl=False, other_options=None):
    assert ((cov_estimation or diag) != identity or cov_estimation == diag == identity == False)

    num_episodes, num_parameters, num_objectives = estimated_gradients.shape[:]
    estimated_gradients = estimated_gradients.transpose((1, 2, 0)).reshape((num_parameters*num_objectives, num_episodes), order='F')
    mu = np.mean(estimated_gradients, axis=1)
    if verbose:
        print('computed mu')
    if other_options is None:
        other_options = [False, False, False, False]
    if other_options[0]:
        # all ones
        sigma = np.ones((len(mu), len(mu)))
    elif other_options[1]:
        # girl
        sigma = np.kron(np.ones((num_objectives, num_objectives)), np.eye(num_parameters))
    elif len(other_options) >= 3 and other_options[2]:
        #block covariance
        if cov_estimation:
            sigma = estimate_cov(estimated_gradients, diag)
        elif diag:
            sigma = np.diag(np.std(estimated_gradients, axis=1) ** 2)
        else:
            sigma = np.cov(estimated_gradients)
        sigma = sigma * np.kron(np.ones((num_objectives, num_objectives)), np.eye(num_parameters)) / num_episodes
    elif len(other_options) > 3 and other_options[3]:
        # averaged block covariance
        # makes the problem convex as in Corollary 4.1
        ones = np.ones((num_objectives, 1))
        if cov_estimation:
            sigma = estimate_cov(estimated_gradients, False)
        else:
            sigma = np.cov(estimated_gradients)
        ones_kron_iden = np.kron(ones, np.eye(num_parameters))
        Q = (1 / num_objectives ** 2) * np.dot(np.dot(ones_kron_iden.T, sigma), ones_kron_iden)
        sigma = np.kron(np.dot(ones, ones.T), Q)
    elif identity:
        sigma = np.eye(len(mu))
    elif cov_estimation:
        sigma = estimate_cov(estimated_gradients, diag) / num_episodes
    elif girl:
        sigma = np.kron(np.ones((num_objectives, num_objectives)), np.eye(num_parameters))
    else:
        if diag:
            sigma = np.diag(np.std(estimated_gradients, axis=1) ** 2) / num_episodes
        else:
            sigma = np.cov(estimated_gradients) / num_episodes

    if verbose:
        print('computed cov')
    identity = np.identity(num_parameters)

    # def rank_approx(w):
    #     w = np.reshape(w, (-1, 1))
    #     kr = np.kron(w, identity)
    #     left_exp = np.dot(kr.T, np.dot(sigma, kr))
    #     if verbose:
    #         print('left exp condition number: ', np.linalg.cond(left_exp))
    #     right_exp = np.linalg.solve(left_exp, np.dot(kr.T, mu))
    #     obj = np.dot(np.dot(mu.T, kr), right_exp)
    #     return obj
    def rank_approx(w):
        #print(w)
        w = np.reshape(w, (-1, 1))
        d = sigma.shape[0] // w.shape[0]

        left_exp = 0.
        right_exp_mu = 0.
        for ii in range(w.shape[0]):
            right_exp_mu += mu[ii*d:(ii+1)*d] * w[ii]
            for jj in range(w.shape[0]):
                left_exp += sigma[ii*d:(ii+1)*d, jj*d:(jj+1)*d] * w[ii] * w[jj]

        if verbose:
            print('left exp condition number: ', np.linalg.cond(left_exp))
        right_exp = np.linalg.solve(left_exp, right_exp_mu)
        obj = np.dot(right_exp_mu.T, right_exp)
        return obj
    obj_func = rank_approx
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bound = (0, 1)
    bounds = [bound] * num_objectives
    if seed is not None:
        np.random.seed(seed)
    evaluations = []
    i = 0
    pbar = tqdm(total=num_iters)
    while i < num_iters:
        x0 = np.random.uniform(0, 1, num_objectives)
        x0 = x0 / np.sum(x0)
        res = optimize.minimize(obj_func,
                                x0,
                                method='SLSQP',
                                constraints=constraint,
                                bounds=bounds,
                                options={'ftol': 1e-8, 'disp': verbose})
        if res.success:
            evaluations.append([res.x, obj_func(res.x)])
            pbar.update(1)
            i += 1
    evaluations = np.array(evaluations)
    min_index = np.argmin(evaluations[:, 1])
    x, y = evaluations[min_index, :]
    pbar.close()
    r = None
    if compute_jacobian:
        vec_r = compute_r(mu, sigma, x, num_parameters)
        r = vec_r.reshape(num_parameters, num_objectives, order='F')
    return x, y, r


def make_loss_function(mu, sigma, identity_mat):

    def compute_loss(w):
        w = np.reshape(w, (-1, 1))
        kr = np.kron(w, identity_mat)
        left_exp = np.dot(kr.T, np.dot(sigma, kr))
        right_exp = np.linalg.solve(left_exp, np.dot(kr.T, mu))
        obj = np.dot(np.dot(mu.T, kr), right_exp)
        return obj
    return compute_loss


def make_weights_assignment_function(mus, sigmas, ids, num_objectives, seed=None, num_iters=10, verbose=False):
    num_agents = len(mus)
    loss_functions = []
    for agent in range(num_agents):
        loss_functions.append(make_loss_function(mus[agent], sigmas[agent], ids[agent]))
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # np.dot(w, w.T) - 1
    bound = (0, 1)
    bounds = [bound] * num_objectives

    def calculate_weights(assignment):

        def rank_approx(w):
            obj = 0
            for i, loss_func in enumerate(loss_functions):
                if assignment[i] > 0:
                    obj += assignment[i] * loss_func(w)
            return obj

        if seed is not None:
            np.random.seed(seed)
        evaluations = []
        i = 0
        #pbar = tqdm(total=num_iters, desc='Weight Assignment')
        while i < num_iters:
            x0 = np.random.uniform(0, 1, num_objectives)
            x0 = x0 / np.sum(x0)
            res = optimize.minimize(rank_approx,
                                    x0,
                                    method='SLSQP',
                                    constraints=constraint,
                                    bounds=bounds,
                                    options={'ftol': 1e-8, 'disp': verbose})
            if res.success:
                evaluations.append([res.x, rank_approx(res.x)])
                #pbar.update(1)
                i += 1
        evaluations = np.array(evaluations)
        min_index = np.argmin(evaluations[:, 1])
        x, y = evaluations[min_index, :]
        #pbar.close()
        return x, y

    return calculate_weights


def solve_cluster_ra_PGIRL(estimated_gradients, sigmas=None, ids=None, cov_estimation=False, diag=False, identity=False,
                           seed=1234, verbose=False, weights=None, num_objectives=None, num_iters=10):
    assert ((cov_estimation or diag) != identity or cov_estimation == diag == identity == False)
    assert sigmas is None or (sigmas is not None and ids is not None)
    #vectorize gradients and calculate cov matrices
    if sigmas is None:
        mus = []
        sigmas = []
        ids = []
        for i, agent_grad in enumerate(estimated_gradients):
            num_episodes, num_parameters, num_objectives = agent_grad.shape[:]
            mu, sigma = estimate_distribution_params(agent_grad, identity=identity, diag=diag,
                                                     cov_estimation=cov_estimation)
            if verbose:
                print('computed mu and sigma')
            id = np.identity(num_parameters)
            mus.append(mu)
            sigmas.append(sigma)
            ids.append(id)
    else:
        mus = estimated_gradients
        assert ids is not None and num_objectives is not None
    if weights is not None:
        assert len(weights) == len(mus)
    else:
        weights = np.ones(len(mus))

    def compute_weighted_distance(w, mu, sigma, id):
        w = np.reshape(w, (-1, 1))
        kr = np.kron(w, id)
        left_exp = np.dot(kr.T, np.dot(sigma, kr))
        if verbose:
            print('left exp condition number: ', np.linalg.cond(left_exp))
        right_exp = np.linalg.solve(left_exp, np.dot(kr.T, mu))
        obj = np.dot(np.dot(mu.T, kr), right_exp)
        return obj

    def rank_approx(w):
        obj = 0
        for i, mu in enumerate(mus):
            if weights[i] > 0:
                obj += weights[i] * compute_weighted_distance(w, mu, sigmas[i], ids[i])
        return obj

    obj_func = rank_approx
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bound = (0, 1)
    bounds = [bound] * num_objectives
    np.random.seed(seed)
    evaluations = []
    i = 0
    while i < num_iters:
        x0 = np.random.uniform(0, 1, num_objectives)
        x0 = x0 / np.sum(x0)
        res = optimize.minimize(obj_func,
                                x0,
                                method='SLSQP',
                                constraints=constraint,
                                bounds=bounds,
                                options={'ftol': 1e-8, 'disp': verbose})
        if res.success:
            evaluations.append([res.x, obj_func(res.x)])
            i += 1
    evaluations = np.array(evaluations)
    min_index = np.argmin(evaluations[:, 1])
    x, y = evaluations[min_index, :]
    return x, y


def solve_lp(null_space):
    n, dim = null_space.shape
    c = np.ones(dim)
    a = - np.array(null_space)
    b = np.zeros(n)
    res = scipy.optimize.linprog(c, a, b, method="simplex", bounds=[(None, None)])
    if res.success:
        return res.x
    else:
        return None


def solve_polyhedra(null_space):
    try:
        import cdd
        A = np.array(null_space)
        b = np.zeros(A.shape[0])

        b = b.reshape((b.shape[0], 1))
        mat = cdd.Matrix(np.hstack([b, -A]), number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
        P = cdd.Polyhedron(mat)
        g = P.get_generators()
        V = np.array(g)[:, 1:]
        num_v = V.shape[0]
        if num_v == 1:
            return None
        else:
            return V[1]
    except ImportError:
        print("Cdd not installed")
        return None


def solve_girl_approx(P, seed=None, verbose=False):
    num_objectives = P.shape[0]
    def quad(w):
        obj = np.dot(np.dot(w.T, P), w)
        return obj
    obj_func = quad
    constraint = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bound = (0, 1)
    bounds = [bound] * num_objectives
    if seed is not None:
        np.random.seed(seed)
    evaluations = []
    i = 0
    while i < 10:
        x0 = np.random.uniform(0, 1, num_objectives)
        x0 = x0 / np.sum(x0)
        res = optimize.minimize(obj_func,
                                x0,
                                method='SLSQP',
                                constraints=constraint,
                                bounds=bounds,
                                options={'ftol': 1e-8, 'disp': verbose})
        if res.success:
            evaluations.append([res.x, obj_func(res.x)])
            i += 1
    evaluations = np.array(evaluations)
    min_index = np.argmin(evaluations[:, 1])
    x, y = evaluations[min_index, :]
    return x, y

##In case Covariance matrix can be written as a Kroneker product
def solve_ra_PGIRL_exact(estimated_gradients, q1=None, q2=None):
    num_episodes, num_parameters, num_objectives = estimated_gradients.shape[:]
    if q1 is None:
        q1 = np.eye(num_objectives)

    if q2 is None:
        q2 = np.eye(num_parameters)

    q1 = scila.sqrtm(q1)
    q2 = scila.sqrtm(q2)

    mean = estimated_gradients.mean(axis=0)
    temp_matrix = np.dot(q2, np.dot(mean, q1))
    u, s, v = np.linalg.svd(temp_matrix, full_matrices=False)
    s[num_objectives-1:] = 0
    s = np.diag(s)
    r = np.dot(q2, np.dot(u, np.dot(s, np.dot(v, q1))))
    ns = linalg.null_space(r)
    ns = ns / np.sum(ns)
    return ns[:, 0]


def compute_r(mu, sigma, w, n):
    w = np.reshape(w, (-1, 1))
    identity = np.eye(n)
    kr = np.kron(w, identity)
    left_exp = np.dot(kr.T, np.dot(sigma, kr))
    right_exp = np.linalg.solve(left_exp, np.dot(kr.T, mu))
    r = mu - np.dot(np.dot(sigma, kr), right_exp)
    return r
