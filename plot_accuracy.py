import numpy as np
from scipy.stats import t
import csv

alpha = 0.98

val = []
val2 = []
val3 = []
val4 = []

load_path = 'data/cont_gridworld_multiple/gpomdp/'
## PLOT RESULTS MLIRL
values = np.load(load_path + 'weights_multiple_sigma_girl.npy', allow_pickle=True)
values2 = np.load(load_path + 'res_mlirl_final13.npy')
for v in values:
    val.append(v[0])
for v in values2:
    val2.append(v[0])
n = len(val[0])
val = np.array(val)
val2 = np.array(val2)

n = [5, 10, 20, 30, 100]
n2 = 20
file = open(load_path + 'clustering_perf.csv', 'w')
writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['n', 'mean', 'high', 'low', 'mean2', 'high2', 'low2'])
for i, n_i in enumerate(n):
    v = val[:, i] / (n_i * 3)
    v2 = val2[:, i] / (n_i * 2)
    s1 = t.interval(alpha, n2 - 1)[1] * np.std(v) / np.sqrt(n2)
    s2 = t.interval(alpha, n2 - 1)[1] * np.std(v2) / np.sqrt(n2)
    me = val[:, i].mean() / (n_i * 3)
    me2 = val2[:, i].mean() / (n_i * 2)
    writer.writerow([n_i, me2, me2 + s2, me2 - s2, me, me + s1, me - s1])
