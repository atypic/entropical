import numpy as np
import numpy as np
import copy
import itertools
import concurrent.futures

#import matplotlib
#matplotlib.use('nbagg')
#import matplotlib.pyplot as plt

def in_grid(state, grid, skip_cells=1):
    l = 1
    for x in reversed(range(len(grid))):
        if np.array_equal(state[::skip_cells], grid[x,::skip_cells]):
            return l
        l += 1

    return -1

def step_ca(timesteps, width, ruleset, startstate):
    cell_grid = np.zeros((timesteps,width), dtype=int)
    g = cell_grid
    #cell_grid[0,int(width/2)] = 1
    cell_grid[0] = startstate #np.random.randint(2, size=width)
    W = width
    #for timesteps
    seen_states = []
    for j in range(timesteps-1):
        for i in range(width):
            #for j, i in itertools.product(range(timesteps-1), range(width)):
            #print(j,i)
            #get the current pattern
            setrule = (g[j, (i - 1) % W]) << 2  |\
                      (g[j, i       % W]) << 1  |\
                      (g[j, (i + 1) % W])
            #print(setrule)
            next_cell_state = (ruleset >> setrule) & 1
            #print(next_cell_state)
            cell_grid[j+1, i] = next_cell_state

        yield cell_grid[j+1]

            #have we seen this state before?
            #for
            #if i == W-1:
            #    x = in_grid(cell_grid[j+1], cell_grid[:j], 1)
            #    if x > -1:
            #        return x, cell_grid
    #return -1, cell_grid

def rollout_ca_para(args):
    init_binary = np.array(tobin(args[1], width))
    trans, grid = rollout_ca(100, args[0], args[2], init_binary)
    decaconfigs = [bool2int(config) for config in grid]
    nums = set()
    print(decaconfig)
    for f in decaconfigs:
        nums.add(f)
    print("uniqe: ", nums)
    return trans, nums


def tobin(x,s):
    return [(x>>k)&1 for k in range(0,s)]

def bool2int(x):
    y = 0
    for i,j in enumerate(x):
        y += j<<i
    return y