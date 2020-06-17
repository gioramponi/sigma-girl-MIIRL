from sklearn import model_selection, linear_model
import numpy as np
import argparse
import time
from trajectories_reader import read_trajectories


NUM_ACTIONS = 4

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='bc_linear')
parser.add_argument('--demonstrations', type=str, default='optimal_policy_1000_trajectories.csv')
args = parser.parse_args()

glob_path = args.dir + '/'
model_path = glob_path + 'models/' + str(time.time()) + '/' + 'logistic'

X_dataset, y_dataset = read_trajectories(glob_path+'demonstrations/'+args.demonstrations)

states = np.array(X_dataset)
actions = np.array(y_dataset)

logistic = linear_model.LogisticRegression(C=10e10, solver='sag', multi_class='multinomial', max_iter=1000, n_jobs=-1,
                                           random_state=12345)
f = logistic.fit(states, actions)
xval = model_selection.cross_val_score(logistic, states, actions, cv=5)
print("Average accuracy = {} +/- {}".format(np.mean(xval), np.std(xval)))
