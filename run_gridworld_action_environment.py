import os
import sys
import numpy as np
import argparse
import csv
import tensorflow as tf
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from envs.continuous_gridword import GridWorldAction
import baselines.common.tf_util as U
from baselines.common.models import mlp
from baselines.common.policies_bc import build_policy
from gym.spaces import Box, Discrete

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# SCRIPT TO COLLECT TRAJECTORIES FOR THE GRIDWORLD ENVIRONMENT


def collect_trajectories(num_trajectories, env, policy, rbf=True):
    trajectories = []
    for i in range(num_trajectories):
        trajectory = []
        s = env.reset(rbf=rbf)
        while True:
            a = policy(s)
            next_s, r, done, info = env.step(a, rbf=rbf)
            trajectory_sample = {
                's': list(s),
                'a': list(a),
                "s'": list(next_s),
                'r': r,
                'features': list(info['features']),
                'done': done
            }
            trajectory.append(trajectory_sample)
            s = list(next_s)
            if done:
                trajectories.append(trajectory)
                break
    return trajectories, trajectory_sample.keys()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_basis', type=int, default=50)
    parser.add_argument("--fast_zone", type=float, default=1.0)
    parser.add_argument("--slow_zone", type=float, default=10.)
    parser.add_argument("--action", type=float, default=5.)
    parser.add_argument("--border_width", type=float, default=1)
    parser.add_argument("--direction", type=str, choices=['center', 'up', 'down', 'border'], default='border')
    parser.add_argument('--model', type=str, default='weights.npy')
    parser.add_argument('--run_policy', action='store_true')
    parser.add_argument('--demonstrations', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug_model', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument("--fail_prob", type=float, default=0.)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--extended_features', action='store_true')
    parser.add_argument('--print_features', action='store_true')
    parser.add_argument('--print_states', action='store_true')
    parser.add_argument('--init_logstd', type=float, default=np.log(np.sqrt(0.1)))
    parser.add_argument('--trainable_variance', action='store_true')

    args = parser.parse_args()
    H = 2 * args.grid_size
    W = args.grid_size
    action_dim = 2

    rew_weights = np.array([args.fast_zone, args.slow_zone, args.action])
    direction = args.direction
    if direction != "center":
        direction = "border"
    env = GridWorldAction(shape=[args.grid_size, args.grid_size], rew_weights=rew_weights,
                    randomized_initial=args.random_start, horizon=args.horizon,
                    n_bases=np.array([W, H]), fail_prob=args.fail_prob,
                    border_width=args.border_width)  # 5, 10
    sum_rew = 0


    def log(message):
        if args.verbose:
            print(message)
    stds = {
        'pi/pi/logstd:0': [[np.log(np.sqrt(0.1)), np.log(np.sqrt(0.1))]],
    }

    net_args = {
        "trainable_variance": args.trainable_variance,
        "init_logstd": args.init_logstd,
    }


    def input_policy(s, stochastic=True):
        command = input()
        print(command)
        try:
            dx, dy = [float(x) for x in command.split(' ')]
        except:
            dx, dy = [0, 0]
        return dx, dy


    policy = input_policy
    if args.run_policy or args.debug_model:
        state_space = np.prod(env.observation_space.shape)
        observation_space = Box(low=-np.inf, high=np.inf, shape=(state_space,))
        action_space = Discrete(action_dim)
        tf.reset_default_graph()
        network = mlp(num_hidden=state_space, num_layers=0)

        policy_train = build_policy(observation_space, action_space, network, train=False, init_bias=0., trainable_bias=False)
        pi = policy_train()
        U.initialize()
        if not args.random:
            log("loading_model")
            theta = np.load(args.model)
            print(theta.shape)
            pi.set_theta(np.ravel(theta))


        def linear_policy():
            #if (np.random.uniform(0, 1) < args.epsilon):
            #    return np.random.randint(0, action_dim)
            s = env.get_state(rbf=True)
            logits, a, state, neglogp = pi.step(s, stochastic=True)
            log("Logits: " + str(logits))
            return a[0]


        policy_label = 'bc'

    policy = linear_policy
    if args.demonstrations > 0:
        trajectories = []
        for i in range(args.demonstrations):
            trajectory = []
            s = env.reset(rbf=True)
            while True:
                a = policy()
                next_s, r, done, info = env.step(a, rbf=True)
                trajectory_sample = {
                    's': list(s),
                    'a': a,
                    "s'": list(next_s),
                    'r': r,
                    'features': list(info['features']),
                    'done': done
                }
                trajectory.append(trajectory_sample)
                s = list(next_s)
                if done:
                    trajectories.append(trajectory)
                    break
        with open('data/' + policy_label + "_" + str(args.demonstrations) + '_trajectories_%s.csv' % args.suffix, 'w+',
                  newline='') as trajectories_file:
            dicti_writer = csv.DictWriter(trajectories_file, fieldnames=trajectory_sample.keys())
            dicti_writer.writeheader()
            for trajectory in trajectories:
                dicti_writer.writerows([sample for sample in trajectory])

    else:
        print("Play!")
        env.reset()
        env._render()
        while True:
            a = policy()
            _, r, done, info = env.step(a)
            if args.debug_model:
                s = env.get_state(rbf=True)
                if 'trpo' in args.model:
                    a, v, state, logits = pi.step(s, stochastic=True)
                else:
                    logits, a, state, neglogp = pi.step(s, stochastic=True)
                log("Logits: " + str(logits))
            sum_rew += r
            # input()
            if args.print_features:
                print("Features:")
                print(info["features"])
            env._render()
            log("Reward :%.2f" % r)
            if done:
                log("Episode finished! Score:%d" % sum_rew)
                log("Press anything to continue!")
                input()
                env.reset()
                sum_rew = 0
                env._render()


