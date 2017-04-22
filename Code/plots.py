
import sys
if "../" not in sys.path:
  sys.path.append("../") 

print sys.getdefaultencoding()
from time import time
from sys import stdout

import numpy as np
import matplotlib.pyplot as plt

from irl import maxent
from irl import deep_maxent
from irl import value_iteration
from irl.mdp.gridworld import Gridworld
from irl.mdp.objectworld import Object
    

def grid_experiment(grid_size, feature_map, epochs, structure, n):

    maxent_data = []
    deep_maxent_data = []
    for n_samples in (32,):
        t = time()
        maxent_EVDs = []
        deep_maxent_EVDs = []
        for i in range(n):
            print("{}: {}/{}".format(n_samples, i+1, n))
            maxent_EVD, deep_maxent_EVD = test_gw_once(grid_size, feature_map,
                                                       n_samples, epochs,
                                                       structure)
            maxent_EVDs.append(maxent_EVD)
            deep_maxent_EVDs.append(deep_maxent_EVD)
            print(maxent_EVD, deep_maxent_EVD)
            stdout.flush()
        maxent_data.append((n_samples, np.mean(maxent_EVDs),
                           np.std(maxent_EVDs)))
        deep_maxent_data.append((n_samples, np.mean(deep_maxent_EVDs),
                                np.std(deep_maxent_EVDs)))
        print("{} (took {:.02}s)".format(n_samples, time() - t))
        print("MaxEnt:", maxent_data)
        print("DeepMaxEnt:", deep_maxent_data)
    return maxent_data, deep_maxent_data


    wind = 0.3
    discount = 0.9
    learning_rate = 0.01
    trajectory_length = 3*grid_size

    plt.subplot(3, 3, 1)
    plt.pcolor(ground_reward.reshape((grid_size, grid_size)))
    plt.title("Groundtruth reward")
    plt.colorbar()
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
                    bottom=False, top=False, left=False, right=False,
                    labelright=False)
    plt.subplot(3, 3, 2)
    plt.pcolor(maxent_reward.reshape((grid_size, grid_size)))
    plt.title("MaxEnt reward")
    plt.colorbar()
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
                    bottom=False, top=False, left=False, right=False,
                    labelright=False)
    plt.subplot(3, 3, 3)
    plt.pcolor(deep_maxent_reward.reshape((grid_size, grid_size)))
    plt.title("DeepMaxEnt reward")
    plt.colorbar()
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
                    bottom=False, top=False, left=False, right=False,
                    labelright=False)

    plt.subplot(3, 3, 4)
    plt.pcolor(optimal_policy.reshape((grid_size, grid_size)), vmin=0, vmax=3)
    plt.title("Optimal policy")
    plt.colorbar()
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
                    bottom=False, top=False, left=False, right=False,
                    labelright=False)
    plt.subplot(3, 3, 5)
    plt.pcolor(maxent_policy.reshape((grid_size, grid_size)), vmin=0, vmax=3)
    plt.title("MaxEnt policy")
    plt.colorbar()
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
                    bottom=False, top=False, left=False, right=False,
                    labelright=False)
    plt.subplot(3, 3, 6)
    plt.pcolor(deep_maxent_policy.reshape((grid_size, grid_size)),
               vmin=0, vmax=3)
    plt.title("DeepMaxEnt policy")
    plt.colorbar()
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
                    bottom=False, top=False, left=False, right=False,
                    labelright=False)

    plt.subplot(3, 3, 7)
    plt.pcolor(optimal_V.reshape((grid_size, grid_size)))
    plt.title("Optimal value")
    plt.colorbar()
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
                    bottom=False, top=False, left=False, right=False,
                    labelright=False)
    plt.subplot(3, 3, 8)
    plt.pcolor(maxent_V.reshape((grid_size, grid_size)))
    plt.title("MaxEnt value")
    plt.colorbar()
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
                    bottom=False, top=False, left=False, right=False,
                    labelright=False)
    plt.subplot(3, 3, 9)
    plt.pcolor(deep_maxent_V.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("DeepMaxEnt value")
    plt.tick_params(labeltop=False, labelbottom=False, labelleft=False,
                    bottom=False, top=False, left=False, right=False,
                    labelright=False)  
    plt.savefig("{}_{}_{}_{}gridworld{}.png".format(grid_size, feature_map,
        n_samples, epochs, structure, np.random.randint(10000000)))

print(grid_experiment(10, "kernel", 1000,(10,10) ,1))