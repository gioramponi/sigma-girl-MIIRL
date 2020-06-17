import os
import sys
import tensorflow as tf
import numpy as np
import math
import argparse
from tqdm import trange
import random
import time
from gym.spaces import Box, Discrete
from trajectories_reader import read_trajectories
path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from baselines.common.policies_bc import build_policy
from baselines.common.models import mlp
import baselines.common.tf_util as U


ACTIONS = ['up', 'right', 'down', 'left']

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=2000)
parser.add_argument('--num_layers', type=int, default=0)
parser.add_argument('--dir', type=str, default='bc_linear')
parser.add_argument('--demonstrations', type=str, default='optimal_policy_1000_trajectories.csv')
parser.add_argument('--validation', type=float, default=0.2)
parser.add_argument('--stochastic_eval', action='store_true')
parser.add_argument('--save_best', action='store_true')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--multiply_left', type=int, default=10)
args = parser.parse_args()

glob_path = args.dir + '/'
model_name = str(args.num_epochs) + '_' + str(args.num_layers)
tf_path = glob_path + 'tensorboards/' + model_name + '_' + str(time.time()) + '/'
model_path = glob_path + 'models/' + str(time.time()) + '/' + model_name

X_dataset, y_dataset = read_trajectories(glob_path+'demonstrations/'+args.demonstrations)

tf.set_random_seed(543210)
np.random.seed(543210)
random.seed(543210)
observation_space = Box(low=-np.inf, high=np.inf, shape=(len(X_dataset[0]), ))
action_space = Discrete(len(ACTIONS))
tf.reset_default_graph()
config = tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=8,
            intra_op_parallelism_threads=8,
            device_count={'CPU': 8}
)
config.gpu_options.allow_growth = True
sess = U.make_session(make_default=True, config=config)
network = mlp(num_hidden=len(X_dataset[0]), num_layers=args.num_layers)
policy_train = build_policy(observation_space, action_space, network, l2=args.l2, lr=args.lr)()
U.initialize()
writer = tf.summary.FileWriter(tf_path)

# dataset build
states = np.array(X_dataset)
actions = np.array(y_dataset)
print(len(states))
left_states = states[actions == 3]
left_actions = actions[actions == 3]
print(len(left_states))
for i in range(args.multiply_left):
    states = np.concatenate((states, left_states))
    actions = np.concatenate((actions, left_actions))
dataset = list(zip(states, actions))
random.seed(49680)
random.shuffle(dataset)
if args.validation > 0.:
    k = math.floor(args.validation * len(dataset))
    dataset_training = dataset[:-k]
    dataset_validation = dataset[-k:]
else:
    dataset_training = dataset[:]

# pre-processing statistics
num_batches = len(dataset_training) // args.batch_size
print('# batches: ', num_batches)
print('# training samples: ', len(dataset_training))
logger = {
    'training_samples': len(dataset_training),
    'batch_size': args.batch_size,
    'num_batches': num_batches,
    'num_epochs': args.num_epochs
}
if args.validation > 0.:
    print('# validation samples: ', len(dataset_validation))
    logger['validation_samples'] = len(dataset_validation)

# validation samples built
X_val, y_val = zip(*dataset_validation)
X_val, y_val = np.array(X_val), np.array(y_val)
XX_val, yy_val = [], []
for i in range(len(ACTIONS)):
    XX_val.append(X_val[y_val == i])
    yy_val.append(y_val[y_val == i])

np.random.seed(654321)
# train + accuracy over epochs
counter = 0
best_accuracy = 0
for epoch in trange(args.num_epochs):
    # train batches built
    random.shuffle(dataset_training)
    batches = []
    for i in range(num_batches):
        base = args.batch_size * i
        batches.append(dataset_training[base: base + args.batch_size])
    # train
    for batch in batches:
        batch_X, batch_y = zip(*batch)
        output = policy_train.fit(batch_X, batch_y)
        summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=output[0]), ])
        writer.add_summary(summary, counter)
        counter += 1
    # validation
    if args.validation > 0.:
        overall_accuracy = 0
        for i in range(len(ACTIONS)):
            accuracy, _ = policy_train.evaluate(XX_val[i], yy_val[i], args.stochastic_eval)
            summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy_"+ACTIONS[i], simple_value=accuracy), ])
            writer.add_summary(summary, epoch)
            overall_accuracy += accuracy
        overall_accuracy /= len(ACTIONS)
        summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy_overall", simple_value=overall_accuracy), ])
        writer.add_summary(summary, epoch)
        if args.save_best and epoch % 10 == 0: #best_accuracy <= overall_accuracy:
            policy_train.save(model_path + '_' + str(epoch))#+ '_best')
            best_accuracy = overall_accuracy

with open(tf_path+'/log.txt', 'w') as log_file:
    log_file.write(str(logger))

sess.close()
