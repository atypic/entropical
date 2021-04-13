"""
    Predictive information

    This is the mutual information between a time series x(t) before a certain point in time, t
    and the information after this point in time.

   Lizier has derived a 'local' variant of this measurement, in which each site
    is regarded locally; and then time series of a certain length k before and after t is considered.

    Basically it is the per-cell local information storage:

    I(xt,xt+) = H(xt) - H(xt|xt+)

    the last one is tricky-- the conditional information: it depends on knowing the full joint distribution, which we
    don't have. But we can fill in the 'table'.

    p(x,y) = how many times have we seen x and y at the same time: count and divide by total number of states.


"""
import numpy as np
import queue


class InformationMeasurements:
    def __init__(self, k=16, j=1, settling_time=60):

        self.distribution_past_states = {}
        self.distribution_next_state = {}
        self.distribution_joint_past_next = {}

        self.joint_past_neighbor = {}
        self.joint_next_past_neighbor = {}

        self.total_observations = 0

        self.k = k
        self.j = j
        self.settling_time = settling_time + self.k

        return

    def add_run(self, next_state_generator):
        ca_state = next(next_state_generator)
        #states = np.zeros((len(ca_state), self.k + 1), dtype=int)
        states = np.zeros((self.k + 1, len(ca_state)), dtype=int)
        states[0, ...] = ca_state
        qpos = 0
        #q = queue.Queue()
        #q.put(ca_state)
        boolint_cache = {}

        self.total_observations = 0
        for t, ca_state in enumerate(next_state_generator):
            print(f"Stepping state generator: {t}")
            states = np.roll(states, 1, axis=0)
            states[0] = ca_state
            if t < self.settling_time:
                continue

            w = len(ca_state)

            # Build the required distributions by maximum likelihood estimation (counting samples :)
            # iterate time dimension so transpose to have time first dim
            # timeseries = states.T
            for cell, timeslice in enumerate(states.T):

                self.total_observations += 1
                #print(f"states: {cell}")
                #assert(len(timeslice) == 17)

                # remember that timeslice is 0 indexed, so when we do timeslice[k] it is 'n+1'.
                """
                k_slice = None
                if set(timeslice[:self.k]) not in boolint_cache.keys():
                    k_slice = self.bool2int(timeslice[:self.k])
                    boolint_cache[set(timeslice[:self.k])] = k_slice
                else:
                    k_slice = boolint_cache[set(timeslice[:self.k])]
                """
                k_slice = timeslice[1:].tostring()
                #print(np.fromstring(k_slice, dtype=np.int))

                xkin = k_slice
                if xkin in self.distribution_past_states.keys():
                    self.distribution_past_states[xkin] += 1
                else:
                    self.distribution_past_states[xkin] = 1

                #distribution of next states
                xin = (timeslice[0])  # , cell, n+1)
                if xin in self.distribution_next_state.keys():
                    self.distribution_next_state[xin] += 1
                else:
                    self.distribution_next_state[xin] = 1

                # joint distribution (both xkin, and xin)
                xin = (k_slice, timeslice[0])
                if xin in self.distribution_joint_past_next.keys():
                    self.distribution_joint_past_next[xin] += 1
                else:
                    self.distribution_joint_past_next[xin] = 1

                # joint between past and neighboring cell, same time (k)
                key = (k_slice, states[1, (cell - self.j) % w])
                if key in self.joint_past_neighbor.keys():
                    self.joint_past_neighbor[key] += 1
                else:
                    self.joint_past_neighbor[key] = 1

                #joint between next, past, and j neighbor
                key = (timeslice[0], k_slice, states[1, (cell - self.j) % w])
                if key in self.joint_next_past_neighbor.keys():
                    self.joint_next_past_neighbor[key] += 1
                else:
                    self.joint_next_past_neighbor[key] = 1

        return self.distribution_past_states

    #The timeslice needs to be such that the last element represents the n+1 element.
    def predictive_information(self, timeslice):
        k_slice = timeslice[1:].tostring()
        term_a = self.p_xk_xnext(k_slice, timeslice[0])
        term_b = self.p_xk(k_slice) * self.p_xnext(timeslice[0])
        print(f"P slice {timeslice[1:]}, next:  {timeslice[0]}, term_a {term_a} term_b: {term_b}")
        return np.log2(term_a) - np.log2(term_b)

    #neighbor cell state at time = n, not n+1.
    def information_transfer(self, timeslice, neigbhor_cell_state):
        k_slice = timeslice[1:].tostring()
        joint_a = self.p_xnext_xk_xj(timeslice[0], k_slice, neigbhor_cell_state)/\
                  self.p_xk_xj(k_slice, neigbhor_cell_state)
        joint_b = self.p_xk_xnext(k_slice, timeslice[0])/self.p_xk(k_slice)

        return np.log2(joint_a) - np.log2(joint_b)

    def p_xk(self, xk):
        #denom = sum(self.distribution_past_states.values())
        denom = self.total_observations
        if not self.distribution_past_states:
            print("Please add_run() first")
        else:
            if xk not in self.distribution_past_states.keys():
                return 0
            return self.distribution_past_states[xk] / denom

    def p_xnext(self, x):
        denom = sum(self.distribution_next_state.values())
        denom = self.total_observations
        if not self.distribution_next_state:
            print("Please add_run() first")
        else:
            if not x in self.distribution_next_state.keys():
                return 0
            else:
                return self.distribution_next_state[x] / denom

    def p_xk_xnext(self, xk, x_next):
        denom = sum(self.distribution_joint_past_next.values())
        denom = self.total_observations
        if not self.distribution_joint_past_next:
            print("Please add_run() first")
        else:
            if (xk, x_next) not in self.distribution_joint_past_next.keys():
                return 0
            else:
                return self.distribution_joint_past_next[(xk, x_next)] / denom

    def p_xk_xj(self, xk, xj):
        denom = sum(self.joint_past_neighbor.values())
        denom = self.total_observations
        if (xk,xj) not in self.joint_past_neighbor.keys():
            return 0
        else:
            return self.joint_past_neighbor[(xk, xj)] / denom

    def p_xnext_xk_xj(self, xnext, xk, xj):
        denom = sum(self.joint_next_past_neighbor.values())
        denom = self.total_observations
        if (xnext, xk, xj) not in self.joint_next_past_neighbor.keys():
            return 0
        else:
            return self.joint_next_past_neighbor[(xnext, xk, xj)] / denom


    @staticmethod
    def tobin(x, s):
        return [(x >> k) & 1 for k in range(0, s)]

    @staticmethod
    def bool2int(x):
        y = 0
        for i, j in enumerate(x):
            y += j << i
        #for b in x:
        #    y = (y << 1) | b
        return y
