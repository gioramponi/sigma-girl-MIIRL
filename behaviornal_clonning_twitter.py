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

path_to_add = os.path.dirname(os.path.abspath(__file__)) + '/..'
sys.path = [path_to_add] + sys.path
from baselines.common.policies_bc import build_policy
from baselines.common.models import mlp
import baselines.common.tf_util as U


ACTIONS = ['nop', 'tweet']

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers of the mlp')
parser.add_argument('--num_hidden', type=int, default=8, help='number of hidden units per layer')
parser.add_argument('--dir', type=str, default='bc', help='directory where to save the logs and trained policies')
parser.add_argument('--agents', type=str, default='', help='comma separated list of agents to clone')
parser.add_argument('--dir_to_read', type=str, default='data_twitter/', help='directory where to '
                                                                             'read the demonstrations')
parser.add_argument('--validation', type=float, default=0.2, help='size of validation set')
parser.add_argument('--stochastic_eval', action='store_true', help='evaluate accuracy with stochastic policy')
parser.add_argument('--save_best', action='store_true', help='save the best policy')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--l2', type=float, default=0., help='l2 regularization hyperparameter')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--ratio', type=int, default=3, help='threshhold on action imbalance to perform augmentation')
parser.add_argument('--seed', type=int, default=543210, help='random seed')
parser.add_argument('--weight_classes', action='store_true', help='use a weightes loss to account for class imbalance')
args = parser.parse_args()

glob_path = args.dir + '/'
model_name = str(args.num_epochs) + '_' + str(args.num_layers)

tf.set_random_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

states_data = np.load(args.dir_to_read + 'real_states.pkl', allow_pickle=True)
actions_data = np.load(args.dir_to_read + 'actions.pkl', allow_pickle=True)
agent_keys = list(actions_data.keys())

if args.agents != '':

    agent_keys = [x for x in args.agents.split(',')]
for agent in agent_keys:
    print("Cloning Agent: " + agent)
    start_time = str(time.time())
    tf_path = glob_path + 'tensorboards/' + agent + '/' + model_name + '_' + start_time + '/'
    model_path = glob_path + 'models/' + agent + '/' + model_name + '_' + start_time + '/'
    X_dataset = states_data[agent]
    y_dataset = actions_data[agent]

    # dataset build
    states = np.array(X_dataset)
    actions = np.array(y_dataset).flatten()
    print(states.shape)
    print(actions.shape)
    tweet_states = states[actions == 1]
    tweet_actions = actions[actions == 1]
    nop_states = states[actions == 0]
    nop_actions = actions[actions == 0]
    num_tweet = (actions == 1).sum()
    num_nop = (actions == 0).sum()
    if num_nop > args.ratio * num_tweet:
        for i in range(min(int((num_nop / num_tweet) // 2), 10)):
            states = np.concatenate((states, tweet_states))
            actions = np.concatenate((actions, tweet_actions))
    elif num_tweet > args.ratio * num_nop:
        for i in range(min(int((num_tweet / num_nop) // 2), 10)):
            states = np.concatenate((states, nop_states))
            actions = np.concatenate((actions, nop_actions))
    num_tweet = (actions == 1).sum()
    num_nop = (actions == 0).sum()
    if args.weight_classes:
        ratio = num_tweet / (num_tweet + num_nop)
        class_weights = [ratio, 1 - ratio]
    else:
        class_weights = None
    dataset = list(zip(states, actions))
    random.shuffle(dataset)

    observation_space = Box(low=-np.inf, high=np.inf, shape=(len(X_dataset[0]), ))
    action_space = Discrete(len(ACTIONS))
    tf.reset_default_graph()
    # config = tf.ConfigProto(
    #             allow_soft_placement=True,
    #             inter_op_parallelism_threads=8,
    #             intra_op_parallelism_threads=8,
    #             device_count={'CPU': 8}
    # )
    #config.gpu_options.allow_growth = True
    sess = U.make_session(make_default=True)#, config=config
    network = mlp(num_hidden=args.num_hidden, num_layers=args.num_layers)
    policy_train = build_policy(observation_space, action_space, network, l2=args.l2, lr=args.lr, train=True,
                                class_weights=class_weights)()
    U.initialize()
    writer = tf.summary.FileWriter(tf_path)


    if args.validation > 0.:
        k = math.floor(args.validation * len(dataset))
        dataset_training = dataset[:-k]
        dataset_validation = dataset[-k:]
    else:
        dataset_training = dataset[:]

    # pre-processing statistics
    num_batches = len(dataset_training) // args.batch_size
    if len(dataset_training) % args.batch_size > 0:
        num_batches +=1
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
        try :
            for batch in batches:
                batch_X, batch_y = zip(*batch)
                output = policy_train.fit(batch_X, batch_y)
                summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=output[0]), ])
                writer.add_summary(summary, counter)
                counter += 1
        except:
            print("Error")
        # validation
        if args.validation > 0.:
            overall_accuracy = 0
            for i in range(len(ACTIONS)):
                try:
                    accuracy, _ = policy_train.evaluate(XX_val[i], yy_val[i], args.stochastic_eval)
                except:
                    print("Error")
                summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy_"+ACTIONS[i], simple_value=accuracy), ])
                writer.add_summary(summary, epoch)
                overall_accuracy += accuracy
            overall_accuracy /= len(ACTIONS)
            summary = tf.Summary(value=[tf.Summary.Value(tag="accuracy_overall", simple_value=overall_accuracy), ])
            writer.add_summary(summary, epoch)
            if args.save_best and epoch % 10 == 0 and best_accuracy <= overall_accuracy:
                policy_train.save(model_path + 'best')
                best_accuracy = overall_accuracy

    with open(tf_path+'/log.txt', 'w') as log_file:
        log_file.write(str(logger))

    sess.close()
