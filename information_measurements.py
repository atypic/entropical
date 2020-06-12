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
        self.timesteps = 600
        self.cells = 40
        self.k = 16

        return

    """ 
    calculate the distribution at timestep t using k steps back.
    """
    def p_marginal(self, k, t):
        """ the probability of seeing the k-step trajectory x(k, i, t).
            the probability of seint a certain "semi infinite past" of k timesteps back.
            ... for each site.

            for each site:
                collect the different k-length trajectories at site i, at time t.
        """

        return

    """ Look k timesteps into the past, compute the 
    
    """
    def build_distribution(self, repeats): #, system, *system_args):
        settling_time = 50 - self.k   #subtract because we also want to fill the queue
        #data = np.array((1, timesteps, cells))
        self.run = {}
        for r in range(repeats):
            q = queue.Queue()
            start_state = np.random.randint(2, size=(self.cells))
            for t, ca_state in enumerate(ca.step_ca(self.timesteps, self.cells, 54, start_state)):
            #for t in range(self.timesteps):
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
                if q.qsize() < self.k:
                    continue   #we haven't filled the queue yet.

                states = np.zeros((self.k, self.cells), dtype=int)
                for idx, state in enumerate(list(q.queue)):
                    for ci, c in enumerate(state):
                        states[idx, ci] = c

                #iterate each 'time dimension'
                for cell, timeslice in enumerate(np.rollaxis(states, 1)):
                    #xkin = (str(timeslice), cell, t)
                    xkin = (ca.bool2int(timeslice), cell, t)

                    if xkin in self.run.keys():
                        self.run[xkin] += 1
                    else:
                        self.run[xkin] = 1
                #pop oldest
                q.get()
            print('done')
            #
        #now we make the xkin distribution
        #for s in run:
        #    print(s, "->", run[s])

        return self.run
    def p_xkin(self, xk, i, n):
        if not self.run:
            print("Please build_distribution() first")
        else:
            return self.run[(xk, i, n)] / (self.timesteps * self.cells * 2**self.k)

    """ This is the 'local information storage' for site i (cell)"""
    def a(self, i, n, k):
        return

    """ local information storage for timestep t, looking k cells back """
    def lis(self, t, k):
        return

#pr = pf.Profile()
#pr.disable()

s = StateDistribution()
r = s.build_distribution(1000)
#for key in r:
    #plt.plot(r[key])
    #print("Prob:", s.p_xkin(*key))
    #p = s.p_xkin(*key)
    #if p > 1.e-9:
        #print(p)
    #    plt.plot(p)
plt.hist([s.p_xkin(*key) for key in r], 10)
plt.savefig('prob.png')
#pr.dump_stats('profile.pstat')
#pr.print_stats()
