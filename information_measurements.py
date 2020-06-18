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
import pandas as pd
import queue
import cProfile as pf
import matplotlib.pyplot as plt

class StateDistribution():
    def __init__(self):

        self.run = None
        self.run_xin1 = None
        self.run_xkin_xin1 = None
        self.timesteps = 4
        self.cells = 10000
        self.k = 16
        self.start_states = []

        return

    def build_distribution(self, repeats): #, system, *system_args):
        settling_time = 50 - self.k   #subtract because we also want to fill the queue
        #data = np.array((1, timesteps, cells))
        self.run = {}
        self.run_xin1 = {}
        self.run_xkin_xin1 = {}
        for r in range(repeats):
            q = queue.Queue()
            start_state = np.random.randint(2, size=(self.cells))
            self.start_states.append(start_state)
            for t, ca_state in enumerate(ca.step_ca(self.timesteps, self.cells, 54, start_state)):
                # iterate system
                #print(ca_state)
                #state = np.zeros((cells, 1))
                if t < settling_time:
                    continue

                #We use a queue and limit the number of elements in the queue to 16.
                #It's a fixed length FIFO: each time we read out a new state this is
                #x_k.
                #This relies on listifying the queue. TODO: bad practice?
                q.put(ca_state)
                if q.qsize() < self.k + 1:    #plus one to account for the prediciton t+1
                    continue   #we haven't filled the queue yet.
                states = np.zeros((self.k + 1, self.cells), dtype=int)
                for time, state in enumerate(list(q.queue)):
                    for ci, c in enumerate(state):
                        states[time, ci] = c
                #pop oldest
                q.get()
                n = t - 1   #we're one timestep behind because we also need the last state.

                #iterate each 'time dimension' so roll the axis over to have time first
                for cell, timeslice in enumerate(states.T):

                    xkin = (ca.bool2int(timeslice[:self.k]), cell, n)
                    if xkin in self.run.keys():
                        self.run[xkin] += 1
                    else:
                        self.run[xkin] = 1

                    xin = (timeslice[self.k], cell, n+1)
                    if xin in self.run_xin1.keys():
                        self.run_xin1[xin] += 1
                    else:
                        self.run_xin1[xin] = 1

                    #joint distribution
                    xin = (ca.bool2int(timeslice[:self.k]), cell, n, timeslice[self.k], cell, n+1)
                    if xin in self.run_xkin_xin1.keys():
                        self.run_xkin_xin1[xin] += 1
                    else:
                        self.run_xkin_xin1[xin] = 1

            print('done with repeat', r)
            #
        #now we make the xkin distribution
        #for s in run:
        #    print(s, "->", run[s])
        #self.total_runs = sum(self.run.values())

        return self.run
    def p_xkin(self, xk, i, n):
        denom = sum(self.run.values()) # (self.timesteps-50 + self.k) * 2**self.k
        if not self.run:
            print("Please build_distribution() first")
        else:
            if (xk, i, n) not in self.run.keys():
                return 0
            return self.run[(xk, i, n)] / denom # ((self.timesteps-50) * 2**self.k)

    def p_xin1(self, x, i, n1):
        denom = sum(self.run_xin1.values()) # (self.timesteps - 50 + self.k) * 2
        if not self.run_xin1:
            print("Please build_distribution() first")
        else:
            if not (x, i, n1) in self.run_xin1.keys():
                return 0
            else:
                return self.run_xin1[(x, i, n1)] / denom

    def p_joint(self, xk, x, i, n):
        denom = sum(self.run_xkin_xin1.values()) #(self.timesteps - 50 + self.k) * (2**self.k) * 2
        if not self.run_xkin_xin1:
            print("Please build_distribution() first")
        else:
            if (xk, i, n, x, i, n + 1) not in self.run_xkin_xin1.keys():
                return 0
            else:
                return self.run_xkin_xin1[(xk, i, n, x, i, n + 1)] / denom # ((self.timesteps-50)**2 * 2**self.k * 2)

    """ This is the 'local information storage' for site i (cell)"""
    def a(self, i, n):
        return

    """ local information storage for timestep t, looking k cells back """
    def lis(self, t, k):
        return

#pr = pf.Profile()
#pr.disable()

s = StateDistribution()
r = s.build_distribution(1)
#for key in r:
    #plt.plot(r[key])
    #print("Prob:", s.p_xkin(*key))
    #p = s.p_xkin(*key)
    #if p > 1.e-9:
        #print(p)
    #    plt.plot(p)
#plt.hist([s.p_xkin(*key) for key in r if s.p_xkin(*key) > 6.5e-10], 10)

#print(sum([s.p_xkin(*key) for key in r]))
#plt.savefig('prob.png')

q = queue.Queue()
k = s.k
#cells = s.cells
cells = 40
settling_time = 50 - k
#start_state = np.random.randint(2, size=(cells))
start_state = s.start_states[0][:cells]
res = np.zeros((s.timesteps, cells))
caroll = np.zeros((s.timesteps, cells))
print(res.shape)
for t, ca_state in enumerate(ca.step_ca(s.timesteps, cells, 54, start_state)):
    #funnily enough we need to do exactly the same as when we are making
    #the distsributions...
    if t < settling_time - s.k:
        continue
    if t > s.timesteps - s.k:
        break

    caroll[t] = ca_state
    # We use a queue and limit the number of elements in the queue to 16.
    # It's a fixed length FIFO: each time we read out a new state this is
    # x_k.
    # This relies on listifying the queue. TODO: bad practice?
    q.put(ca_state)
    if q.qsize() < k + 1:  # plus one to account for the prediciton t+1
        continue  # we haven't filled the queue yet.
    states = np.zeros((k + 1, cells), dtype=int)
    for idx, state in enumerate(list(q.queue)):
        for ci, c in enumerate(state):
            states[idx, ci] = c
    q.get()

    for cell, timeslice in enumerate(states.T):
        ain = np.log2(s.p_joint(ca.bool2int(timeslice[:k]), timeslice[k], cell, t)) - np.log2((s.p_xkin(ca.bool2int(timeslice[:k]), cell, t) * s.p_xin1(timeslice[k], cell, t+1)))
        res[t, cell] = ain

for r in res:
    print(r)
posinfo = np.zeros(res.shape)
neginfo = np.zeros(res.shape)
for idx, x in np.ndenumerate(res):
    if x > 0:
        posinfo[idx] = x
    else:
        neginfo[idx] = x
f, axs = plt.subplots(1, 3)
axs[0].matshow(caroll)
axs[1].matshow(posinfo)
axs[2].matshow(neginfo)
plt.matshow(res[settling_time+k:, ...])
plt.show()
