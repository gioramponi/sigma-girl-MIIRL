import pandas as pd
import numpy as np


def read_trajectories(file, all_columns=False, fill_size=0, fill_left=0, cont_actions=False, fix_goal=True, action_size=2):
    if '.csv' not in file:
        return None
    df = pd.read_csv(file)
    states = []
    actions = []
    next_states = []
    rewards = []
    features = []
    dones = []
    episode_count = 0
    ts_count = 0
    for index, row in df.iterrows():
        states.append([float(x) for x in row.loc['s'][1:-1].replace(' ', '').split(',')])
        if cont_actions:
            actions.append([float(x) for x in row.loc['a'][1:-1].replace(' ', '').split(',')])
        else:
            actions.append(row.loc['a'])
        next_states.append([float(x) for x in row.loc["s'"][1:-1].replace(' ', '').split(',')])
        rewards.append(row.loc['r'])
        features.append([float(x) for x in row.loc['features'][1:-1].replace(' ', '').split(',')])
        dones.append(int(row.loc['done']))
        ts_count += 1
        if int(row.loc['done']) == 1 and fix_goal:
            features[-1][2] = 1.
        if int(row.loc['done']) == 1 and fill_size > 0:
            episode_count += 1
            while ts_count % fill_size != 0:
                states.append([float(x) for x in row.loc['s'][1:-1].replace(' ', '').split(',')])
                if cont_actions:
                    actions.append(np.random.uniform(low=-1, high=1, size=action_size))
                else:
                    actions.append(np.random.randint(0, 4))
                next_states.append([float(x) for x in row.loc["s'"][1:-1].replace(' ', '').split(',')])
                rewards.append(0)
                features.append([0 for x in row.loc['features'][1:-1].replace(' ', '').split(',')])
                dones.append(0)
                ts_count += 1
            ts_count = 0
    if fill_left > 0:  # FIX
        for i in range(fill_left):
            s = np.eye(len(states[0]))[np.random.choice(list(range(13, 20)))]
            states.append(s)
            actions.append(3)
    if all_columns:
        return states, actions, next_states, rewards, features, dones
    else:
        return states, actions
