import numpy as np
import sys
from env import taxi
import h5py

# import matplotlib.pyplot as plt
# from collections import deque

from util.colors import bcolors

episodes = int(sys.argv[1])

f = h5py.File("data/qTable.hdf5", "r")
qTable = np.array(f["default"])
f.close()

env = taxi.Env()

totalSteps, totalPenalties = 0, 0

# epiX = [x for x in range(0, episodes)]
# rewardY = deque()
# penaltiesY = deque()

actionDic = {
    0: "DOWN",
    1: "UP",
    2: "RIGHT",
    3: "LEFT",
    4: "PICK UP",
    5: "DROP OFF",
}

for episode in range(episodes):
    steps, penalties = 0, 0
    # totalreward = 0
    state = env.reset()

    done = False
    while not done:
        env.render(1)
        action = np.argmax(qTable[state])
        state, reward, done, _ = env.step(action)

        if reward == -10:
            penalties += 1
        steps += 1

        print("ACTION = " + bcolors.OKGREEN + actionDic[action])
        print(bcolors.ENDC)
        print(
            "CURR_STATE = {}, REWARD = {} SUCCESS_DROP = {}".format(
                state,
                reward,
                done,
            )
        )
        # totalreward += reward

    # rewardY.append(totalreward)
    totalPenalties += penalties
    totalSteps += steps

# plt.plot(epiX, rewardY)
# plt.xlabel("episodes")
# plt.ylabel("reward")
# plt.title("episode vs reward")
# plt.savefig("episodeVSrewardsTest.png")

avgTime = totalSteps / float(episodes)
avgPenalties = totalPenalties / float(episodes)
print("\n\nEvaluation results after {} trials".format(episodes))
print("Average time steps taken: {}".format(avgTime))
print("Average number of penalties incurred: {}".format(avgPenalties))
