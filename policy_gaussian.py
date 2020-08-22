
import numpy as np
from time import sleep
import pickle
import argparse



def create_batch_trajectories(env, batch_size, len_trajectories, param, variance, render=False):
    states = np.zeros((batch_size, len_trajectories, 25))
    actions = np.zeros((batch_size, len_trajectories, 2))
    rewards = np.zeros((batch_size, len_trajectories))
    reward_features = np.zeros((batch_size, len_trajectories, 3))
    goal = 0
    states_ = np.zeros((batch_size, len_trajectories, 2))
    # param = param.reshape((25,2))
    for batch in range(batch_size):
        state = env.reset(rbf=True)
        done = False
        for t in range(len_trajectories):
            action = np.random.multivariate_normal(np.dot(param.T,state), np.eye(2) * variance)
            if not done:
                next_state, reward, done, info, st = env.step(action, rbf=True)
            if render:
                env._render()
                sleep(0.1)
            states[batch, t] = state
            actions[batch, t] = action
            rewards[batch, t] = reward
            reward_features[batch, t] = info['features'][:3]
            states_[batch, t] = st
            if done:
                rewards[batch, t+1:] = 0
                actions[batch, t+1:] = np.zeros(2)
                reward_features[batch, t+1:] = info['features'][:3]
                states_[batch, t+1:] = st
                break
            state = next_state


    return states, actions, rewards, reward_features, states_



def gradient_est(param, batch_size, len_trajectories, states, actions, var_policy):
    gradients = np.zeros((batch_size, len_trajectories, 25, 2))
    for b in range(batch_size):
        for t in range(len_trajectories):
            action = actions[b,t]
            if np.isnan(action[0]):
                gradients[b,t,:] = np.zeros_like(gradients[b,t,:])
            else:
                state = np.array(states[b,t])
                gradients[b,t,:] = (((action - np.dot(param.T, state)).reshape(-1,1) * np.array([state,state])).T / var_policy)
    return gradients


def gpomdp(env, num_batch, batch_size, len_trajectories, initial_param, gamma, var_policy):
    param = np.array(initial_param)
    results = np.zeros(num_batch)
    discount_factor_timestep = np.power(gamma*np.ones(len_trajectories), range(len_trajectories))
    gradient = np.zeros_like(param)
    rewards__ = np.zeros(num_batch)
    gradients__ = np.zeros(num_batch)

    from estimators.gradient_descent import Adam
    optimizer = Adam(learning_rate=0.1, ascent=True)
    optimizer.initialize(param)
    for i in range(num_batch):
        if i > 0:
            param = optimizer.update(gradient)
            # param += 0.05*gradient
        states, actions, rewards,_ ,_ = create_batch_trajectories(env, batch_size, len_trajectories, param, var_policy)
        # print(rewards.shape)
        discounted_return = discount_factor_timestep[np.newaxis, :, np.newaxis] * rewards[:,:,np.newaxis]  # (N,T,L)
        gradients = gradient_est(param, batch_size, len_trajectories, states, actions, var_policy)  # (N,T,K, 2)
        gradient_est_timestep = np.cumsum(gradients, axis=1)  # (N,T,K, 2)
        baseline_den = np.mean(gradient_est_timestep ** 2 + 1.e-10, axis=0)  # (T,K, 2)
        baseline_num = np.mean(
            (gradient_est_timestep ** 2)[:, :, :, :,np.newaxis] * discounted_return[:, :, np.newaxis,np.newaxis, :],
            axis=0)  # (T,K,2,L)

        baseline = baseline_num / baseline_den[:, :, :,np.newaxis]  # (T,K,2,L)


        gradient = np.mean(np.sum(gradient_est_timestep[:, :, :, :,np.newaxis] * (discounted_return[:, :, np.newaxis,np.newaxis, :] -
                                                                            baseline[np.newaxis, :, :]), axis=1),
                           axis=0)  # (K,2,L)

        gradient=np.reshape(gradient,param.shape)
        gradients__[i] = np.linalg.norm(gradient.ravel())
        rewards__[i] = np.mean(np.sum(rewards, axis=1))

    return param, results, states, rewards__, gradients__


def train_policy(args, direction):
    env = continuous_gridworld2.GridWorld(randomized_initial=True, direction=direction, fail_prob=0.)
    param,_,_,_,_ = gpomdp(env, 100, 100, 30, np.random.random((25,2)),args.gamma, args.var)
    return param

if __name__ == '__main__':
    from envs import continuous_gridworld2
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-trajectories', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--len-trajectories', type=int, default=30)
    parser.add_argument('--file', type=str)
    parser.add_argument('--gamma', type=float, default=0.999)
    parser.add_argument('--var', type=float, default=0.1)
    parser.add_argument('--train-policy', type=bool, default=False)



    args = parser.parse_args()

    for t in ['center','border', 'up', 'down', 'center']:
        if not args.train_policy:
            with open("param_policy_%s.pkl" % t, "rb") as f:
                param_policy = pickle.load(f)
        else:
            param_policy = train_policy(args, t)

        gamma = args.gamma
        var_policy = args.var
        env = continuous_gridworld2.GridWorld(randomized_initial=False, direction=t, fail_prob=0.)
        for i in range(args.num_trajectories):
            url = args.file
            try:
                os.makedirs(url + '/dataset_'+str(i))
            except:
                print('created')
            url = url + '/dataset_'+str(i)
            states, actions, _, rewards, st = create_batch_trajectories(env, args.batch_size, args.len_trajectories, param_policy, var_policy, False)
            with open(url+'/trajectories.pkl', 'wb') as f:
                pickle.dump([states, actions, rewards, st], f)

