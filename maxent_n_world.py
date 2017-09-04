"""
Run maximum entropy inverse reinforcement learning on the gridworld MDP.
Matthew Alger, 2015
matthew.alger@anu.edu.au
"""
import time
import numpy as np
import matplotlib.pyplot as plt

import maxent
import n_world

def main(grid_size, dimensions, discount, n_trajectories, epochs, learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.
    Plots the reward function.
    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """
    wind = 0.1
    nw = n_world.nworld(grid_size, dimensions, wind, discount, "parsed.txt")
    trajectories = nw.cluster_grid(14)#number of coordinates to parse
    print("Dimensions: " + str(dimensions))
    print("Grid Size: " + str(grid_size))
    start = time.time()
    feature_matrix = nw.feature_matrix()
    r = maxent.irl(feature_matrix, nw.n_actions, discount,
        nw.transition_probability, trajectories, epochs, learning_rate)
    end = time.time()
    print(end-start)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()
if __name__ == '__main__':
    main(10, 2, 0.01, 20, 200, 0.01)
