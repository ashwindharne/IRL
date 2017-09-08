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
def drawArrow(A, B):
    plt.arrow(A[0], A[1], B[0]-A[0], B[1]-A[1], head_width=0.125, head_length=0.125)
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
    wind = 0.01
    nw = n_world.nworld(grid_size, dimensions, wind, discount, "parsed.txt")
    trajectories = nw.cluster_grid(2, 7)#number of coordinates to parse, trajectory length
    nw.generate_action(trajectories)
    print("Dimensions: " + str(dimensions))
    print("Grid Size: " + str(grid_size))
    start = time.time()
    feature_matrix = nw.feature_matrix()
    r = maxent.irl(feature_matrix, nw.n_actions, discount,
        nw.transition_probability, trajectories, epochs, learning_rate)
    end = time.time()
    print(end-start)
    r = r.reshape(grid_size, grid_size)
    values=np.linspace(0, grid_size-1, grid_size)
    X, Y = np.meshgrid(values, values)
    levels = np.linspace(-1, 1, 40)
    plt.contourf(X, Y, r, levels=levels)

    for trajectory in trajectories:
        for i, point in enumerate(trajectory):
            if i>1:
                 drawArrow(nw.int_to_point(trajectory[i-1][0]), nw.int_to_point(point[0]))
    plt.colorbar()
    plt.show()
if __name__ == '__main__':
    main(5, 2, 0.1, 20, 200, 0.01)

def drawArrow(A, B):
    plt.arrow(A[0], A[1], B[0]-A[0], B[1]-A[1], head_width=0.125, head_length=0.125)
