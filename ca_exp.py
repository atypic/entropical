import information_measurements as im
import numpy as np
import queue
import ca
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

if __name__ == "__main__":

    ruleset = 54

    # Build a distribution
    s = im.InformationMeasurements(k=16)
    sampling_cells = 10000
    sampling_timesteps = 600
    sampling_runs = 1
    for rep in range(sampling_runs):
        initial_state = np.random.randint(2, size=sampling_cells)
        state_generator = ca.step_ca(sampling_timesteps, sampling_cells, ruleset, initial_state)
        s.add_run(state_generator)

    # Distributions ready, let's apply this to a sample system.
    # The system is 40 cells wide, runs for 200 time steps.
    k = s.k
    cells = 37
    timesteps = 1000
    settling_time = 30 + k
    #settling_time = 0
    initial_state = np.random.randint(2, size=cells)
    #initial_state = np.array([1,1,0,1,1,1,1,1,1,0,1,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,1,1,1,0,0,0,1,0,0,0])
    res = np.zeros((timesteps, cells))
    transfer_res = np.zeros((timesteps, cells))
    step_func = ca.step_ca(timesteps, cells, ruleset, initial_state)
    ca_unroll = np.zeros((timesteps, cells))

    ca_state = next(step_func)
    states = np.zeros((k + 1, len(ca_state)), dtype=int)
    states[0, ...] = ca_state
    print(s)

    # Use a queue to store the last k iterations.
    q = queue.Queue()
    for t, ca_state in enumerate(step_func):
        states = np.roll(states, 1, axis=0)
        states[0] = ca_state

        if t < settling_time:
            continue

        ca_unroll[t] = ca_state

        timeseries = states.T
        for cell, ts in enumerate(states.T):
            res[t, cell] = s.predictive_information(ts)
            transfer_res[t, cell] = s.information_transfer(ts, states[k-1, (cell-s.j) % cells])

    # Makin' graphs
    posinfo = np.zeros(res.shape)
    it_posinfo = np.zeros(res.shape)
    neginfo = np.zeros(res.shape)
    it_neginfo = np.zeros(res.shape)
    for idx, x in np.ndenumerate(res):
        if x > 0:
            posinfo[idx] = x
        else:
            neginfo[idx] = x

    for idx, x in np.ndenumerate(transfer_res):
        if x > 0:
            it_posinfo[idx] = x
        else:
            it_neginfo[idx] = x

    #if repeat_ts:
    #    plot_timestep = repeat_ts
    #else:
    plot_timestep = 150
    plot_size = 200

    f, axs = plt.subplots(2, 3)
    f.suptitle(f"Rule {ruleset}")
    ims = []
    ims.append(axs[0,0].matshow(ca_unroll[plot_timestep:plot_timestep+plot_size], cmap=plt.get_cmap('gray').reversed()))
    ims.append(axs[0,1].matshow(posinfo[plot_timestep:plot_timestep+plot_size], cmap=plt.get_cmap('gray').reversed()))
    ims.append(axs[0,2].matshow(neginfo[plot_timestep:plot_timestep+plot_size], cmap=plt.get_cmap('gray')))
    ims.append(axs[1,0].matshow(ca_unroll[plot_timestep:plot_timestep+plot_size], cmap=plt.get_cmap('gray').reversed()))
    ims.append(axs[1,1].matshow(it_posinfo[plot_timestep:plot_timestep+plot_size], cmap=plt.get_cmap('gray').reversed()))
    ims.append(axs[1,2].matshow(it_neginfo[plot_timestep:plot_timestep+plot_size], cmap=plt.get_cmap('gray')))
    axs[0,0].set_title('System rollout')
    axs[0,1].set_title('Positive information storage')
    axs[0,2].set_title('Negative information storage')
    axs[1,0].set_title('Autobots roll out')
    axs[1,1].set_title('Positive information tx, left to right')
    axs[1,2].set_title('Negative information tx, left to right')
    for iax, ax in np.ndenumerate(axs):
        div = make_axes_locatable(ax).append_axes("right", size="5%", pad=0.05)
        f.colorbar(ims[iax[0] * 3 + iax[1]], cax=div)
    plt.show()

