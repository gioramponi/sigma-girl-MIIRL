import os
import sys
import numpy as np
import argparse
import csv
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from envs.lqg import LQG
from policies.linear_gaussian_policy import LinearGaussianPolicy

# SCRIPT TO COLLECT TRAJECTORIES FOR THE LQG ENVIRONMENT
parser = argparse.ArgumentParser()
parser.add_argument('--dimension', type=int, default=2, help='number of dimensions of the state and action space')
parser.add_argument('--state_noise', type=float, default=0.3, help='noise of the transition kernel')
parser.add_argument('--action_noise', type=float, default=0.3, help='noise in the policy')
parser.add_argument('--diag_state_noise', default='0.0,0.0', type=str, help='vector of coma separated values to specify'
                                                                            ' more complex noise models')
parser.add_argument('--diag_action_noise', default='0.1,1', type=str, help='vector of coma separated values to specify'
                                                                            ' more complex noise models')
parser.add_argument('--a', default='1,0.2,0.2,1', type=str, help='parameter of the reward')
parser.add_argument('--b', default='1,0.2,0.2,1', type=str, help='parameter of the reward')
parser.add_argument('--action_scales', default='0.02, 0.05', type=str, help='Induce non-symmetrical effect'
                                                                            ' of the actions')
parser.add_argument('--max_state', type=float, default=5, help='maximum state')
parser.add_argument('--max_action', type=float, default=np.inf, help='maximum action')
parser.add_argument('--horizon', type=int, default=40, help='horizon of the task')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--seed', type=float, default=1, help='random seed')
parser.add_argument('--shift', type=float, default=0.5, help='make the reward function non-uniform')
parser.add_argument('--run_policy', action='store_true', help='flag to signal wether we want to execute a given'
                                                              ' policy or input the actions manually')
parser.add_argument('--demonstrations', type=int, default=0, help='Number of episodes to collect and save in'
                                                                  ' datasets')
parser.add_argument('--verbose', action='store_true', help='enable_logging')
parser.add_argument('--random_init', action='store_true', help='start the episodes in random positions')
parser.add_argument('--random', action='store_true', help='execute random policy')
parser.add_argument('--interactive', action='store_true', help='used when demonstrations is 0,'
                                                               'stop execution at each time step')
parser.add_argument('--print_states', action='store_true', help='log state features')
parser.add_argument('--save_policy', action='store_true', help='save the weights of the optimal policy')
parser.add_argument('--diff_scales', action='store_true', help='impact the initial state probabilities')

args = parser.parse_args()
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
    if len(a) > args.dimension **2:
        a = a[:args.dimension**2]
    a = np.array(a).reshape((args.dimension, args.dimension))

if args.b == '':
    b = np.eye(args.dimension)
else:
    b = [float(x) for x in args.b.split(',')]
    if len(b) > args.dimension **2:
        b = b[:args.dimension**2]
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


def log(message):
    if args.verbose:
        print(message)


def input_policy(s):
    command = input()
    try:
        a = [float(x) for x in command.split(' ')]
    except:
        a = [0] * args.dimension
    return np.array(a).reshape((args.dimension, 1))


policy = input_policy
if args.run_policy:
    pi = LinearGaussianPolicy()
    if args.random:
        pi.set_weights(np.random.rand((args.dimension, args.dimension)))
    else:
        K = env.computeOptimalK()
        print("optimal K", K)
        pi.set_weights(K, action_noise)
    policy_label = 'linear_gaussian_policy'
    policy = lambda s: pi.act(s)

else:
    policy_label = 'input_policy'
    policy = input_policy

if args.demonstrations > 0:
    trajectories = []
    for i in range(args.demonstrations):
        trajectory = []
        s = env.reset(random_init=args.random_init, different_scales=args.diff_scales)
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
            s = next_s
            trajectory.append(trajectory_sample)
            if done:
                trajectories.append(trajectory)
                break
    out_dir = 'data/lqg/' + policy_label + "/"
    if args.save_policy:
        K = env.computeOptimalK()
        np.save(out_dir + 'optimal_lqg_policy.npy', K)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(out_dir + str(args.demonstrations)+'_trajectories.csv', 'w+',
              newline='') as trajectories_file:
        dicti_writer = csv.DictWriter(trajectories_file, fieldnames=trajectory_sample.keys())
        dicti_writer.writeheader()
        for trajectory in trajectories:
            dicti_writer.writerows([sample for sample in trajectory])
else:
    print("Play!")
    sum_rew = 0
    s = env.reset(random_init=args.random_init, different_scales=args.diff_scales)
    env._render()
    while True:
        if args.print_states:
            print("state:", s)
        a = policy(s)
        print("Action:", a)

        s, r, done, info = env.step(a)
        sum_rew += r
        if args.interactive:
            input()
        env._render()
        log("Reward :%.2f" % r)
        if done:
            log("Episode finished! Score:%d" % sum_rew)
            log("Press anything to continue!")
            input()
            s = env.reset(random_init=args.random_init, different_scales=args.diff_scales)
            sum_rew = 0
            env._render()
