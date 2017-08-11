"""
Run large state space linear programming inverse reinforcement learning on the
gridworld MDP.
Matthew Alger, 2015
matthew.alger@anu.edu.au
"""
import time
import numpy as np
import matplotlib.pyplot as plt

import linear_irl
import n_world
from value_iteration import value

def main(grid_size, discount, dimensions):
    """
    Run large state space linear programming inverse reinforcement learning on
    the gridworld MDP.
    Plots the reward function.
    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    """
    start = time.time()
    wind = 1
    trajectory_length = 4*grid_size
    nw = n_world.nworld(grid_size, dimensions, wind, discount)

    ground_r = np.array([nw.reward(s) for s in range(nw.n_states)])
    policy = [nw.optimal_policy_deterministic(s) for s in range(nw.n_states)]

    # Need a value function for each basis function.
    feature_matrix = nw.feature_matrix()
    values = []
    for dim in range(feature_matrix.shape[1]):
        reward = feature_matrix[:, dim]
        values.append(value(policy, nw.n_states, nw.transition_probability,
                            reward, nw.discount))
    values = np.array(values)

    r = linear_irl.large_irl(values, nw.transition_probability,
                        feature_matrix, nw.n_states, nw.n_actions, policy)
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
    r = np.split(r, 5)
    plt.subplot(2, 2, 1)
    plt.pcolor(r[0].reshape(grid_size, grid_size))
    plt.colorbar()
    plt.title("z = 0")
    plt.subplot(2,2,2)
    plt.pcolor(r[1].reshape(grid_size, grid_size))
    plt.colorbar()
    plt.title("z = 1")
    plt.subplot(2, 2, 3)
    plt.pcolor(r[2].reshape(grid_size, grid_size))
    plt.colorbar()
    plt.title("z = 2")
    plt.subplot(2, 2, 4)
    plt.pcolor(r[3].reshape(grid_size, grid_size))
    plt.colorbar()
    plt.title("z = 3")
    plt.subplot(2, 2, 4)
    plt.pcolor(r[4].reshape(grid_size, grid_size))
    plt.colorbar()
    plt.title("z = 4")
    plt.tight_layout()
    plt.show()
    """
    
if __name__ == '__main__':
    main(10, 0.9, 2)
