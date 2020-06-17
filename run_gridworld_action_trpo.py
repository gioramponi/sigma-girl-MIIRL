import os
import sys
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import numpy as np
from mpi4py import MPI
from baselines import logger
from baselines.trpo_mpi import trpo_mpi
from envs.grid_word_AB import GridWorld
from envs.continuous_gridword import GridWorldAction
from baselines.common.models import mlp
import baselines.common.tf_util as U
from policies.eval_policy import eval_and_render_policy
import time
import argparse
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# SCRIPT TO TRAIN A TRPO AGENT IN THE GRIDWORLD ENVIRONMENT

def train_trpo(num_timesteps, eval_episodes, seed, horizon, out_dir='.',
               load_path=None, checkpoint_path_in=None,
               gamma=0.99, grid_size=5, first_zone=-1.0, second_zone=-10., action=2.,
               timesteps_per_batch=500,rand_initial=True, clip_mean=False,
               direction='border', fail_prob=0.1, border_width=2.,
               continuous=True, n_basis=None, num_layers=0, num_hidden=32,
               checkpoint_freq=20, init_logstd=-1, trainable_variance=False, trainable_bias=False):
    if n_basis is None:
        #n_basis = np.array([grid_size, 2 * grid_size])
        n_basis = np.array([2*grid_size, 4 * grid_size])
    start_time = time.time()
    clip = None
    if clip_mean:
        clip = (-5, 5)
    rew_wights = [first_zone, second_zone, action]
    print(rew_wights)
    print(fail_prob)
    print(horizon)
    if continuous:
        dir = 'cont_gridworld'
        env = GridWorldAction(shape=[grid_size, grid_size], rew_weights=rew_wights, horizon=horizon,
                            randomized_initial=rand_initial, fail_prob=fail_prob, border_width=border_width,
                            n_bases=n_basis, direction=direction)
        env_eval = GridWorldAction(shape=[grid_size, grid_size], rew_weights=rew_wights, horizon=horizon,
                                 randomized_initial=rand_initial, fail_prob=fail_prob, border_width=border_width,
                                 n_bases=n_basis, direction=direction)
    else:
        dir = 'gridworld'
        env = GridWorld(gamma=gamma, rew_weights=rew_wights, fail_prob=fail_prob, horizon=horizon,
                        shape=(grid_size, grid_size), randomized_initial=rand_initial, direction=direction)
        env_eval = GridWorld(gamma=gamma, rew_weights=rew_wights, fail_prob=fail_prob, horizon=horizon,
                             shape=(grid_size, grid_size), randomized_initial=rand_initial, direction=direction)

    directory_output = (dir + '/trpo-rews-' + str(first_zone) + '_' + str(second_zone) + '_'
                        + str(action)) + '/' + direction

    def eval_policy_closure(**args):
        return eval_and_render_policy(env_eval, **args)

    tf.set_random_seed(seed)
    sess = U.single_threaded_session()
    sess.__enter__()
    rank = MPI.COMM_WORLD.Get_rank()
    time_str = str(start_time)
    if rank == 0:
        logger.configure(dir=out_dir + '/' + directory_output + '/logs',
                         format_strs=['stdout', 'csv'], suffix=time_str)
    else:
        logger.configure(format_strs=[])
        logger.set_level(logger.DISABLED)

    network = mlp(num_hidden=num_hidden, num_layers=num_layers)
    trpo_mpi.learn(network=network, env=env, eval_policy=eval_policy_closure, timesteps_per_batch=timesteps_per_batch,
                   max_kl=0.001, cg_iters=10, cg_damping=1e-3,
                   total_timesteps=num_timesteps, gamma=gamma, lam=1.0, vf_iters=3, vf_stepsize=1e-4,
                   checkpoint_freq=checkpoint_freq,
                   checkpoint_dir_out=out_dir + '/' + directory_output + '/models/'+time_str+'/',
                   load_path=load_path, checkpoint_path_in=checkpoint_path_in,
                   eval_episodes=eval_episodes,
                   init_logstd=init_logstd,
                   trainable_variance=trainable_variance,
                   trainable_bias=trainable_bias,
                   clip=None)
    print('TOTAL TIME:', time.time() - start_time)
    print("Time taken: %f seg" % ((time.time() - start_time)))
    print("Time taken: %f hours" % ((time.time() - start_time)/3600))

    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_layers', type=int, default=0, help='number of hidden layers of mlp')
    parser.add_argument('--num_hidden', type=int, default=32, help='number of hidden units per layer')
    parser.add_argument('--grid_size', type=int, default=7, help='size of the grid')
    parser.add_argument('--timesteps_per_batch', type=int, default=500, help='batch size for gradient computation')
    parser.add_argument('--n_basis', default='', type=str, help='number of basis in the rbf state representation')
    parser.add_argument("--fast_zone", type=float, default=1., help='reward weight')
    parser.add_argument("--slow_zone", type=float, default=10., help='reward weight')
    parser.add_argument("--action", type=float, default=2., help='reward weight')
    parser.add_argument("--border_width", type=float, default=1., help='size of the border region')
    parser.add_argument("--init_logstd", type=float, default=-3, help='initial variance of the policy')
    parser.add_argument("--fail_prob", type=float, default=0.1, help='state transition noise')
    parser.add_argument('--gamma', type=float, default=0.9999, help='discount factor')
    parser.add_argument('--rand_initial', action='store_true', help='randomness in the initial state distribution')
    parser.add_argument('--continuous', action='store_true', help='continuous environment')
    parser.add_argument('--direction', type=str, choices='center,up,down,border', default='border',
                        help='type of agent to train')
    parser.add_argument('--clip_mean', action='store_true', help='clip policy mean')
    parser.add_argument('--trainable_variance', action='store_true', help='fit the variance of the policy')
    parser.add_argument('--trainable_bias', action='store_true', help='fit the bias of the policy')
    parser.add_argument('--checkpoint_freq', type=int, default=20, help='frequency of chekcpoints of the policy model'
                                                                        ' during training')
    parser.add_argument("--max_timesteps", type=int, default=250000,
                         help='Maximum number of timesteps')
    parser.add_argument("--eval_episodes", type=int, default=100,
                         help='Episodes of evaluation')
    parser.add_argument("--seed", type=int, default=8,
                        help='Random seed')
    parser.add_argument('--horizon', type=int, help='horizon length for episode',
                        default=50)
    parser.add_argument('--dir', help='directory where to save data',
                        default='logs/')
    parser.add_argument('--load_path', help='directory where to load model',
                        default='')
    parser.add_argument('--checkpoint_path_in', help='directory where to load model',
                        default='')

    args = parser.parse_args()

    if args.n_basis == '':
        n_basis = None
    else:
        n_basis = [int(x) for x in args.n_basis.split(',')]

    out_dir = args.dir
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if args.load_path == "":
        args.load_path = None
    if args.checkpoint_path_in == "":
        args.checkpoint_path_in = None

    train_trpo(args.max_timesteps, args.eval_episodes,  args.seed, args.horizon, out_dir, args.load_path,
               args.checkpoint_path_in, args.gamma, args.grid_size, args.fast_zone, args.slow_zone, args.action,
                args.timesteps_per_batch, args.rand_initial, args.clip_mean, args.direction, args.fail_prob,
               args.border_width, args.continuous, n_basis=n_basis, num_layers=args.num_layers,
               num_hidden=args.num_hidden, checkpoint_freq=args.checkpoint_freq, init_logstd=args.init_logstd,
               trainable_variance=args.trainable_variance, trainable_bias=args.trainable_variance)
