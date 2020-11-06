# Import libraries
from numba import njit, prange
from pylab import *

# Define length of transient
Ns = 50000

# Width/Height of grid
width = 128
height = 128
# Initial distribution of trees
initProb = 0.6
# Enum for states
empty, tree, fire, burnt = [int(i) for i in range(4)]

# Parameters for theta
lightning_chance = 0.0002
regrowth_chance = 0.001

# Setup grid
config = zeros([height, width], np.int8)
# Setup grid with trees
for x in range(width):
    for y in range(height):
        if random() < initProb:
            state = tree
        else:
            state = empty
        config[y, x] = state

# Setup the next config variable
nextConfig = zeros([height, width], np.int8)


@njit
def cellUpdate(config, nextConfig, random_numbers,neighbors):
    global empty, tree, fire, burnt
    # Iterate through the whole grid
    for x in range(width):
        for y in range(height):
            # Get current state
            state = config[y, x]
            # If the state is fire then set empty
            if state == fire:
                state = burnt
            # If state is empty then set to tree
            if state == empty or state == burnt:
                # Check if random number is less than the chance
                if random_numbers[x, y] < regrowth_chance:
                    # Setup a variable to hold if fire is close so we don't back on ourselves
                    localFire = False
                    # Iterate through neighbours
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            # Don't wrap around
                            if y + dy >= height or x + dx >= width or dx * dy is not 0:
                                continue
                            # Check if fire
                            if config[y + dy, x + dx] == fire:
                                # If so set that we have seen fire in neighbourhood
                                localFire = True
                    # If no local fire then continue with tree growth
                    if not localFire:
                        state = tree
            # Check if we are a tree
            elif state == tree:
                # Iterate through neighbours
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        # Don't wrap around
                        if y + dy >= height or x + dx >= width or dx * dy is not 0:
                            continue
                        # Check if we can percolate (ensuring that we are using the neighbors lookup of prob)
                        if config[y + dy, x + dx] == fire and random_numbers[x, y] <= neighbors[dx+2, dy+2]:
                            # Set our state to fire
                            state = fire
                # Check if we got struck by lightning and we aren't a fire from above
                if random_numbers[x, y] < lightning_chance and state != fire:
                    # Set us as fire
                    state = fire
            # Set the next config to the state we just got
            nextConfig[y, x] = state


@njit(parallel=True)
def updating(config, nextConfig, alpha):
    global empty, tree, fire, burnt

    # Set alpha fixed to ignition
    ignition = 0.5
    # Calculate dispersion from alpha
    dispersion = ignition * alpha

    # Neighbourhood calculations
    neighbors = array(
        [[0, 0, dispersion * ignition, 0, 0],
         [0, ignition * dispersion ** 0.5, ignition, ignition * dispersion ** 0.5, 0],
         [ignition * dispersion ** 0.5, ignition, 0, ignition, ignition * dispersion ** 0.5],
         [0, ignition * dispersion ** 0.5, ignition, ignition * dispersion ** 0.5, 0],
         [0, 0, dispersion * ignition, 0, 0]]
    )

    # Storage of the transient
    transient = zeros((Ns,4))
    # Iterate over time
    for _ in prange(Ns):
        # Generate randoms for this time value
        random_numbers = np.random.random_sample((width, height))
        # Run cell update
        cellUpdate(config, nextConfig, random_numbers,neighbors)
        # Switch next over to current
        config, nextConfig = nextConfig, config
        # Get data from current state into transient
        transient[_] = np.array(
            [count_nonzero(config == empty),
             count_nonzero(config == tree),
             count_nonzero(config == fire),
             count_nonzero(config == burnt)]
        )
    # Return back the transient response
    return transient


# Calculate the transient
transient = updating(config, nextConfig, 0.02)
# Plot the whole transient
plot(transient)
legend(["Empty","Tree","Fire","Burnt"])
axes = gca()
axes.set_xlabel("t")
axes.set_ylabel("# of cells")
show()

# Define a sweeping of alpha
alpha_sweep = [0.2, 0.02, 0.002, 0.0002]
# Define holding array
alpha_data = []

# Iterate over alpha_sweep
for i in alpha_sweep:
    # Get current transient
    transient = updating(config, nextConfig, i)
    # Get fire + burnt cells
    burning = transient[:, 2] + transient[:, 3]
    # Append last 5000 (stable)
    alpha_data.append(average(burning[-5000:, ]))

# Plot the alpha_data on a log log plot
plot(alpha_sweep,alpha_data)
axes = gca()
axes.set_xscale("log")
axes.set_xlabel("alpha")
axes.set_ylabel("Average burning")
title("theta="+str(lightning_chance/regrowth_chance))
show()
