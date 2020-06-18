import os
import sys
import time
import numpy as np
import argparse
import csv
import tensorflow as tf
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from envs.continuous_gridword import GridWorld
from baselines.common.models import mlp
from utils import load_policy

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


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
    parser.add_argument('--grid_size', type=int, default=7)
    parser.add_argument('--horizon', type=int, default=100)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--n_basis', type=int, default=50)
    parser.add_argument("--fast_zone", type=float, default=1.0)
    parser.add_argument("--slow_zone", type=float, default=10)
    parser.add_argument("--goal", type=float, default=0.)
    parser.add_argument("--border", type=float, default=2)
    parser.add_argument("--border_width", type=float, default=1.5)
    parser.add_argument("--direction", type=str, choices=['center', 'up', 'down', 'border'], default='border')
    parser.add_argument('--model', type=str, default='weights.npy')
    parser.add_argument('--run_policy', action='store_true')
    parser.add_argument('--demonstrations', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--debug_model', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument("--fail_prob", type=float, default=0.1)
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--random_start', action='store_true')
    parser.add_argument('--extended_features', action='store_true')
    parser.add_argument('--print_features', action='store_true')
    parser.add_argument('--print_states', action='store_true')
    parser.add_argument('--init_logstd', type=float, default=-2)
    parser.add_argument('--clip_mean', action='store_true')
    parser.add_argument('--trainable_variance', action='store_true')

    args = parser.parse_args()
    H = 2 * args.grid_size
    W = args.grid_size
    action_dim = 2
    clip = None
    if args.clip_mean:
        clip = (-1, 1)
    rew_weights = np.array([args.fast_zone, args.slow_zone, args.goal, args.border])
    direction = args.direction
    if direction != "center":
        direction = "border"
    env = GridWorld(shape=[args.grid_size, args.grid_size], rew_weights=rew_weights,
                    randomized_initial=args.random_start, horizon=args.horizon,
                    n_bases=np.array([W, H]),
                    direction=args.direction, fail_prob=args.fail_prob,
                    border_width=args.border_width)  # 5, 10
    sum_rew = 0


    def log(message):
        if args.verbose:
            print(message)
    stds = {
        'pi/pi/logstd:0': [[-0.7, 0.7]],
    }

    net_args = {
        "trainable_variance": args.trainable_variance,
        "init_logstd": args.init_logstd,
        "clip": clip
    }


    def input_policy(s, stochastic=True):
        command = input()
        try:
            dx, dy = [float(x) for x in command.split(' ')]
        except:
            dx, dy = [0, 0]
        return dx, dy


    policy = input_policy
    if args.run_policy or args.debug_model:

        tf.reset_default_graph()
        network = mlp(num_hidden=32, num_layers=0)
        linear = 'gpomdp' in args.model
        X_dim = W * H
        pi = load_policy(X_dim=X_dim, model=args.model, continuous=True, num_actions=2, n_bases=X_dim,
                         trainable_variance=args.trainable_variance, init_logstd=args.init_logstd,
                         linear=False)
        pi.load(args.model)


        def linear_policy(s):
            # s = env.get_state(rbf=True)
            logits, a, state, neglogp = pi.step(s, stochastic=True, logits=True)
            log("Logits: " + str(logits))
            return a[0]


        policy_label = 'trpo'
        policy = linear_policy

    elif not args.run_policy:
        policy = input_policy
        policy_label = 'input_policy'

    if args.demonstrations > 0:
        trajectories, data_keys = collect_trajectories(args.demonstrations, env, policy, rbf=True)
        out_dir = "/".join(args.model.split('/')[:-1])
        out_dir += "/trajectories/" + str(time.time()) + "/"
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        with open(out_dir + str(args.demonstrations)+'_trajectories.csv', 'w+',
                  newline='') as trajectories_file:
            dicti_writer = csv.DictWriter(trajectories_file, fieldnames=data_keys)
            dicti_writer.writeheader()
            for trajectory in trajectories:
                dicti_writer.writerows([sample for sample in trajectory])
    else:
        print("Play!")
        s = env.reset(rbf=True)
        env._render()
        while True:
            a = policy(s)
            print("Action:", a)
            if args.debug_model:
                s = env.get_state(rbf=True)
                if 'trpo' in args.model:
                    mean, std = pi._evaluate([pi.pd.mean, pi.pd.std], s, True)

                    log("(mean - std): " + str(mean) + ";" + str(std))
            s, r, done, info = env.step(a, rbf=True)


            sum_rew += r
            #input()
            if args.interactive:
                input()
            env._render()
            if args.print_features:
                print("Features:")
                print(info["features"])
            if args.print_states:
                s = env.get_state()
                print("state:")
                print(s)
            log("Reward :%.2f" % r)
            if done:
                log("Episode finished! Score:%d" % sum_rew)
                log("Press anything to continue!")
                input()
                s = env.reset(rbf=True)
                sum_rew = 0
                env._render()
