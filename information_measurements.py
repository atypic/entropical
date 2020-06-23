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
import ca
import numpy as np
import queue
import matplotlib.pyplot as plt


class InformationMeasurements:
    def __init__(self, k=16, settling_time=30):

        self.distribution_past_states = {}
        self.distribution_next_state = {}
        self.distribution_joint_past_next = {}

        self.k = k
        self.settling_time = settling_time

        return

    def add_run(self, next_state_generator):
        settling_time = self.settling_time + self.k  # want to fill the queue
        q = queue.Queue()
        for t, ca_state in enumerate(
                next_state_generator):
            if t < settling_time:
                continue

            """
             We use a queue and limit the number of elements in the queue to 16.
             It's a fixed length FIFO: each time we read out a new state this is x_k.
             This relies on listifying the queue which is probably not good.
            """

            q.put(ca_state)
            if q.qsize() < self.k + 1:  # plus one to account for the prediciton t+1
                continue  # we haven't filled the queue yet.

            states = np.zeros((self.k + 1, len(ca_state)), dtype=int)
            for time, state in enumerate(list(q.queue)):
                for ci, c in enumerate(state):
                    states[time, ci] = c

            # pop oldest
            q.get()
            n = t - 1  # n is now the 'current state'

            # iterate time dimension so transpose to have time first dim
            for cell, timeslice in enumerate(states.T):

                xkin = (ca.bool2int(timeslice[:self.k]))
                if xkin in self.distribution_past_states.keys():
                    self.distribution_past_states[xkin] += 1
                else:
                    self.distribution_past_states[xkin] = 1

                xin = (timeslice[self.k])  # , cell, n+1)
                if xin in self.distribution_next_state.keys():
                    self.distribution_next_state[xin] += 1
                else:
                    self.distribution_next_state[xin] = 1

                # joint distribution
                xin = (ca.bool2int(timeslice[:self.k]), timeslice[self.k])
                if xin in self.distribution_joint_past_next.keys():
                    self.distribution_joint_past_next[xin] += 1
                else:
                    self.distribution_joint_past_next[xin] = 1

        return self.distribution_past_states

    def p_xk(self, xk):
        denom = sum(self.distribution_past_states.values())
        if not self.distribution_past_states:
            print("Please add_run() first")
        else:
            if xk not in self.distribution_past_states.keys():
                return 0
            return self.distribution_past_states[xk] / denom

    def p_xnext(self, x):
        denom = sum(self.distribution_next_state.values())
        if not self.distribution_next_state:
            print("Please add_run() first")
        else:
            if not x in self.distribution_next_state.keys():
                return 0
            else:
                return self.distribution_next_state[x] / denom

    def p_xk_xnext(self, xk, x):
        denom = sum(self.distribution_joint_past_next.values())
        if not self.distribution_joint_past_next:
            print("Please add_run() first")
        else:
            if (xk, x) not in self.distribution_joint_past_next.keys():
                return 0
            else:
                return self.distribution_joint_past_next[(xk, x)] / denom


if __name__ == "__main__":

    # Build a distribution
    s = InformationMeasurements(k=16)
    cells = 10000
    sampling_runs = 1
    for rep in range(sampling_runs):
        initial_state = np.random.randint(2, size=cells)
        state_generator = ca.step_ca(1000, cells, 110, initial_state)
        s.add_run(state_generator)

    # Distributions should be ready, let's apply this to a sample system.
    # The system is 40 cells wide, runs for 200 timesteps.
    k = s.k
    cells = 40
    timesteps = 200
    settling_time = 30 + k
    initial_state = np.random.randint(2, size=cells)
    res = np.zeros((timesteps, cells))
    step_func = ca.step_ca(timesteps, cells, 110, initial_state)
    ca_unroll = np.zeros((timesteps, cells))

    q = queue.Queue()
    for t, ca_state in enumerate(step_func):
        # funnily enough we need to do exactly the same as when we are making
        # the distsributions...
        if t < settling_time:
            continue
        if t > timesteps:
            break

        ca_unroll[t] = ca_state
        q.put(ca_state)
        if q.qsize() < k + 1:  # plus one to account for the prediciton t+1
            continue  # we haven't filled the queue yet.
        states = np.zeros((k + 1, cells), dtype=int)
        for idx, state in enumerate(list(q.queue)):
            for ci, c in enumerate(state):
                states[idx, ci] = c
        q.get()

        for cell, timeslice in enumerate(states.T):
            ain = np.log2(s.p_xk_xnext(ca.bool2int(timeslice[:k]), timeslice[k])) - \
                  np.log2((s.p_xk(ca.bool2int(timeslice[:k])) * s.p_xnext(timeslice[k])))
            res[t, cell] = ain

    posinfo = np.zeros(res.shape)
    neginfo = np.zeros(res.shape)
    for idx, x in np.ndenumerate(res):
        if x > 0:
            posinfo[idx] = x
        else:
            neginfo[idx] = x
    f, axs = plt.subplots(1, 3)
    a0 = axs[0].matshow(ca_unroll[100:150], cmap=plt.get_cmap('gray'))
    a1 = axs[1].matshow(posinfo[100:150], cmap=plt.get_cmap('gray'))
    a2 = axs[2].matshow(neginfo[100:150], cmap=plt.get_cmap('gray'))
    f.colorbar(a0, ax=axs[0])
    f.colorbar(a1, ax=axs[1])
    f.colorbar(a2, ax=axs[2])
    # plt.matshow(res[settling_time+k:, ...])
    plt.show()
