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
    wind = 0.3
    nw = n_world.nworld(grid_size, dimensions, wind, discount, "parsed.txt")
    nw.cluster_grid()
    #trajectories = nw.parse_trajectories()
    #actions = nw.cluster_actions(trajectories)
    #print(actions)
    """
    print("Dimensions: " + str(dimensions))
    print("Grid Size: " + str(grid_size))
    start = time.time()
    wind = 0.3
    trajectory_length = 3*grid_size
    nw = n_world.nworld(grid_size, dimensions, wind, discount)
    trajectories = nw.generate_trajectories(n_trajectories,
                                            trajectory_length,
                                            nw.optimal_policy)
    #trajectories = nw.parse_trajectories("parsed.txt", 2)
    feature_matrix = nw.feature_matrix()
    ground_r = np.array([nw.reward(s) for s in range(nw.n_states)])
    r = maxent.irl(feature_matrix, nw.n_actions, discount,
        nw.transition_probability, trajectories, epochs, learning_rate)
    end = time.time()
    print(end-start)
    plt.subplot(1, 2, 1)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Groundtruth reward")
    plt.subplot(1, 2, 2)
    plt.pcolor(r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()
    """
if __name__ == '__main__':
    main(5, 2, 0.01, 20, 200, 0.01)
