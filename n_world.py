from sklearn.cluster import KMeans
import numpy as np
import numpy.random as rn
from minisom import MiniSom
import matplotlib.pyplot as plt

class nworld(object):
    """
    N-dimensional gridworld MDP.
    """
    def __init__(self, grid_size, dimensions, wind, discount, file):
        """
        grid_size: Grid size. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        dimensions: number of dimensions. int.
        -> nworld
        """
        #setting member variables of the nworld
        self.file = file
        self.n_states = grid_size**(dimensions)
        self.grid_size = grid_size
        self.wind = wind
        self.discount = discount
        self.dimensions = dimensions
        #generates the corresponding set of incremental actions, e.g. (1,0)
        #default action set, can be changed
        self.actions = []
        for i in range (dimensions):
            action = [(0) for _ in range (dimensions)]
            action1 = [(0) for a in range (dimensions)]
            action[i] = 1
            action1[i] = -1
            action = tuple(action)
            action1 = tuple(action1)
            self.actions.append(action)
            self.actions.append(action1)
        self.actions = tuple(self.actions)
        self.n_actions = len(self.actions)
        # Preconstruct the transition probability array.
        # Gets far too large in higher dimensions/grid sizes
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])
    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,
                                              self.discount)

    def feature_vector(self, i, feature_map="ident"):
        """
        Get the feature vector associated with a state integer.
        i: State int.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> Feature vector.
        """

        if feature_map == "coord":
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f
    def feature_matrix(self, feature_map="ident"):
        """
        Get the feature matrix for this gridworld.
        feature_map: Which feature map to use (default ident). String in {ident,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        """
        Convert a state int into the corresponding coordinate.
        i: State int.
        Iterator n is the dimension of each coordinate
        """
        pt = []
        for n in range (self.dimensions):
            pt.append((i%((self.grid_size)**(n+1)))//((self.grid_size)**n))
        return tuple(pt)
    def point_to_int(self, p):
        """
        Convert a coordinate into the corresponding state int.
        p: (x, y) tuple.
        -> State int.
        """
        s_int = 0
        for cd in range (len(p)):
            s_int += p[cd] * ((self.grid_size)**cd)
        return s_int

    def act(self, s, a):
        """
        TRIVIAL HELPER FUNCTION
        returns state s1 that occurs when action a is performed from state s
        s -> point(tuple)
        a -> action(tuple)
        s1 -> point(tuple)
        """
        s1 = []
        for n in range (self.dimensions):
            s1.append(s[n] + a[n])
            if s1[n] < 0 or s1[n] > self.grid_size-1:
                return s
        return tuple(s1)
    def neighbouring(self, i, k):
        """
        TRIVIAL HELPER FUNCTION
        Get whether two points neighbour each other. Also returns true if they
        are the same point. The logic is slightly more complex in higher
        dimensions with differing action sets, as we must not check the
        absolute distance from point to point, but rather whether there is a
        valid action that can take us from state i to k.
        i:  state, int tuple.
        k:  state, int tuple.
        -> bool.
        """
        if(i == k):
            return True
        for n in range (len(self.actions)):
            if(self.intended(i, self.actions[n], k)):
                return True
        return False
    def intended(self, i, a, j):
        """
        TRIVIAL HELPER FUNCTION
        Get whether j is the intended final state if performing action a
        from state i
        i -> point(tuple)
        a -> action(tuple)
        j -> point(tuple)
        """
        for l in range (self.dimensions):
            if (i[l] + a[l]) != (j[l]):
                return False
        return True
    def corner(self, p):
        """
        TRIVIAL HELPER FUNCTION
        returns whether or not point p is a corner of the world
        p -> point(tuple)
        """
        for i in range (len(p)):
            if (p[i] != 0) and (p[i] != (self.grid_size)-1):
                return False
        return True
    def edge(self, p):
        """
        TRIVIAL HELPER FUNCTION
        returns whether or not point p is on the edge of the world
        p -> point(tuple)
        """
        for i in range (len(self.actions)):
            if (self.off_grid(p, self.actions[i])):
                return True
        return False
    def off_grid(self, p, a):
        """
        TRIVIAL HELPER FUNCTION
        returns whether the action a from state p would move the agent off the
        grid
        p -> point(tuple)
        a -> action(tuple)
        """
        sn = self.act(p, a)
        for n in range (len(p)):
            if(sn[n] != p[n]):
                return False
        return True
    def _transition_probability(self, i, a, j):
        """
        Get the probability of transitioning from state i to state k given
        action j.
        i: State int.
        a: Action int.
        j: State int.
        -> p(s_j | s_i, a_a)
        """
        ti = self.int_to_point(i)#tuple form of state i
        ta = self.actions[a]#tuple form of action a
        tj = self.int_to_point(j)

        if not self.neighbouring(ti, tj):
            return 0.0

        # Is k the intended state to move to?
        if self.intended(ti, ta, tj):
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.
        if ti != tj:
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (self.corner(ti)):
            # Corner.
            # Can move off the edge in n directions.
            # Did we intend to move off the grid?
            if (self.off_grid(ti, ta)):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + (self.dimensions)*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return (self.dimensions)*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if not (self.edge(ti)):
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if (self.off_grid(ti, ta)):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions

    def cluster_grid(self, num_coords, trajectory_length):
        points = self.get_coords(num_coords)
        som = MiniSom(self.grid_size, self.grid_size, num_coords, learning_rate=0.3)
        som.train_random(points, 300)
        trajectories = []
        trajectory = []
        f = open(self.file)
        st = f.readline()
        i = 0
        while len(st) != 0:
            if st != '\n':
                trajectory.append(points[i])
                i += 1
            elif st == '\n':
                trajectories.append(trajectory)
                trajectory = []
            st = f.readline()
        realtrajectories=[]
        for trajectory in trajectories:
            realtrajectory=[]
            for i in range (0, trajectory_length):
                if(i >= len(trajectory)):
                    realtrajectory.append(realtrajectory[len(realtrajectory)-1])
                    continue
                newpoint = (self.point_to_int(som.winner(trajectory[i])), 0, 0)
                realtrajectory.append(newpoint)
            realtrajectories.append(realtrajectory)
        return(np.array(realtrajectories))
    def generate_action(self, trajectories):
        self.actions=[]
        for trajectory in trajectories:
            for i in range (1, len(trajectory)):
                new_action=tuple(np.subtract(self.int_to_point(trajectory[i][0]), self.int_to_point(trajectory[i-1][0])))
                if new_action in self.actions:
                    continue
                self.actions.append(new_action)
        self.actions=tuple(self.actions)


    def get_coords(self, num_coords):
        points = []
        f = open(self.file)
        st = "default"
        while len(st) != 0:
            st = f.readline()
            means = []
            stds = []
            for n in range (0, num_coords):
                means.append(self.get_mean(self.file, n))
                stds.append(self.get_mean(self.file, n))
            if st != '\n':
                while len(st) > 1:
                    newCoord = st.split()
                    while len(newCoord) > num_coords:
                        newCoord.pop()
                    for i in range(len(newCoord)):
                        newCoord[i] = (float(newCoord[i])-means[i])/stds[i] + 1
                    points.append(newCoord)
                    st = f.readline()
        return np.array(points)
    def get_mean(self, file, coord):
        f = open(file)
        st = "default"
        vals = []
        while len(st) != 0:
            st = f.readline()
            if st != "\n":
                while len(st) > 1:
                    newCoord = st.split()
                    while len(newCoord) > (coord+1):#number of variables used
                        newCoord.pop()
                    vals.append(newCoord[coord])
                    st = f.readline()
        vals = [float(i) for i in vals]
        vals = np.array(vals)
        return np.mean(vals)
    def get_std(self, file, coord):
        f = open(file)
        st = "default"
        vals = []
        while len(st) != 0:
            st = f.readline()
            if st != "\n":
                while len(st) > 1:
                    newCoord = st.split()
                    while len(newCoord) > (coord+1):#number of variables used
                        newCoord.pop()
                    vals.append(newCoord[coord])
                    st = f.readline()
        vals = [float(i) for i in vals]
        vals = np.array(vals)
        return np.std(vals)
