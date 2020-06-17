import numpy as np
import time


def eval_policy(env, pi, n_episodes, verbose=True, interactive=False):

    rewards = []
    logs = []

    for i in range(n_episodes):

        start = time.time()
        s = env.reset(ohe=True)
        t = 0
        rew = 0
        while True:
            a = pi(s)
            ns, r, done, inf = env.step(a[0], ohe=True)
            s = ns
            if interactive:
                print("Action=%d" % a)
                print("Reward=%f" % r)
                input()
            rew += r
            t += 1
            if done:
                break

        if verbose:
            print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(i, rew, t, time.time() - start))
        rewards.append(rew)
        logs.append({"reward": rew})

    avg = np.mean(rewards)
    std = np.std(rewards)
    if verbose:
        print("Average Return = {0} +- {1}".format(avg, std))

    env.reset()

    return avg, std, logs

def eval_and_render_policy(env, pi, n_episodes, verbose=True, interactive=False):

    rewards = []
    logs = []

    n_episodes_render = 1
    horizon = 200

    for i in range(n_episodes):

        start = time.time()
        s = env.reset(ohe=True)
        t = 0
        rew = 0
        if i < n_episodes_render and t < horizon:
            env._render()
            time.sleep(0.1)
        while True:
            a = pi(s)
            ns, r, done, inf = env.step(a[0], ohe=True)
            if i < n_episodes_render and t < horizon:
                env._render()
                time.sleep(0.1)
            s = ns
            if interactive or (i < n_episodes_render and t < horizon):
                print("Action=%s" % a)
                print("Reward=%s" % r)
                #input()
            rew += r
            t += 1
            if done:
                break

        if verbose:
            print("Episode {0}: Return = {1}, Duration = {2}, Time = {3} s".format(i, rew, t, time.time() - start))
        rewards.append(rew)
        logs.append({"reward": rew})

    avg = np.mean(rewards)
    std = np.std(rewards)
    if verbose:
        print("Average Return = {0} +- {1}".format(avg, std))

    env.reset()

    return avg, std, logs