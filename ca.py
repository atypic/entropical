import numpy as np


def step_ca(timesteps, width, ruleset, startstate):
    cell_grid = np.zeros((timesteps, width), dtype=int)
    g = cell_grid
    cell_grid[0] = startstate  # np.random.randint(2, size=width)
    W = width
    for j in range(timesteps - 1):
        for i in range(width):
            setrule = (g[j, (i - 1) % W]) << 2 | \
                      (g[j, i % W]) << 1 | \
                      (g[j, (i + 1) % W])
            next_cell_state = (ruleset >> setrule) & 1
            cell_grid[j + 1, i] = next_cell_state

        yield cell_grid[j + 1]


if __name__ == "__main__":
    w = 50
    t = 100
    initial = np.zeros(w)
    initial[0] = 1
    initial[1] = 1
    initial[2] = 1
    initial[25] = 1
    initial[47] = 1
    initial[48] = 1
    initial[49] = 1
    toprint = [initial]
    for time, state in enumerate(step_ca(t, w, 54, initial)):
        toprint.append(state)

    import matplotlib.pyplot as plt

    plt.matshow(toprint)
