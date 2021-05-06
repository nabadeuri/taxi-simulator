import h5py
from env import taxi
import random
import numpy as np
import sys


# import matplotlib.pyplot as plt
# from collections import deque

episodes = int(sys.argv[1])
alpha = 0.5  # the learning rate
gamma = 0.8  # the discount factor, it quantifies how much importance we give for future rewards
epsilon = 0.6  # exploration and exploitation rate

decay = 0.00001

env = taxi.Env()

qTable = np.zeros([env.num_states, env.num_actions])

# episodeX = [x for x in range(0, episodes)]
# rewardY = deque()
# penaltiesY = deque()

for episode in range(episodes):
    totalPenalties, totalRewards, epochs = 0, 0, 0
    epsilon -= decay

    state = env.reset()

    done = False

    while not done:
        # env.render(60)

        if random.uniform(0, 1) < epsilon:
            action = env.get_action()
        else:
            action = np.argmax(qTable[state])

        nextState, reward, done, _ = env.step(action)

        oldQValue = qTable[state][action]

        prevState = state

        nextMax = np.max(qTable[nextState])

        newQval = (1 - alpha) * oldQValue + alpha * (reward + gamma * nextMax)

        qTable[state][action] = newQval

        if reward == -10:
            totalPenalties += 1

        totalRewards += 1

        epochs += 1

        state = nextState

        print("\nTraining episode {}".format(episode + 1))
        print("Time steps: {}, Penalties: {}".format(epochs, totalPenalties))

    # rewardY.append(totalRewards)
    # penaltiesY.appen(totalPenalties)


# plt.plot(episodeX, rewardY)
# plt.xlabel("episodes")
# plt.ylabel("reward")
# plt.title("episode vs reward")
# plt.savefig("episodeVSreward.png")

with h5py.File("data/qTable.hdf5", "w") as f:
    f.create_dataset("default", data=qTable)

print("done")
