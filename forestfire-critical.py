# Import libraries
from numba import njit, prange
from pylab import *

# Define length of transient
Ns = 163800

# Width/Height of grid
width = 128
height = 128
# Initial distribution of trees
initProb = 0.6
# Enum for states
empty, tree, fire = [int(i) for i in range(3)]

# Neighbourhood calculations
dispersion = 0.001
ignition = 0.5
neighbors = array(
    [[0, 0, dispersion * ignition, 0, 0],
     [0, ignition * dispersion ** 0.5, ignition, ignition * dispersion ** 0.5, 0],
     [ignition * dispersion ** 0.5, ignition, 0, ignition, ignition * dispersion ** 0.5],
     [0, ignition * dispersion ** 0.5, ignition, ignition * dispersion ** 0.5, 0],
     [0, 0, dispersion * ignition, 0, 0]]
)

# Parameters for theta
lightning_chance = 0.000002
regrowth_chance = 0.001

# Define storage of area of fires
fire_area = zeros(Ns, dtype=np.int32)
# Keep track of the fire in grid
fire_data = full((width, height), -1, np.int8)
# Current index (using array to pass by reference - numba is strange with this)
fire_num = array([0])

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
def cellUpdate(config, nextConfig, random_numbers, fire_area, fire_data, fire_num):
    # Iterate through the whole grid
    for x in range(width):
        for y in range(height):
            # Get current state
            state = config[y, x]
            # If the state is fire then set empty
            if state == fire:
                state = empty
            # If state is empty then set to tree
            if state == empty:
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
                            # Get fire number of one will ignite us
                            curr_fire_num = fire_data[y + dy, x + dx]
                            # Increase this fire number area
                            fire_area[curr_fire_num] += 1
                            # Set ourselves to that fire number
                            fire_data[x, y] = curr_fire_num
                            # Set our state to fire
                            state = fire
                # Check if we got struck by lightning and we aren't a fire from above
                if random_numbers[x, y] < lightning_chance and state != fire:
                    # Set us as fire
                    state = fire
                    # Set the fire area to one as we started a fire
                    fire_area[fire_num[0]] = 1
                    # Setup the fire data with our fire
                    fire_data[x, y] = fire_num[0]
                    # Increase the index of fire
                    fire_num[0] += 1
            # Set the next config to the state we just got
            nextConfig[y, x] = state


@njit(parallel=True)
def updating(config, nextConfig, fire_area, fire_data, fire_num):
    # Iterate over time
    for _ in prange(Ns):
        # Setup range of random numbers to sample from
        random_numbers = np.random.random_sample((width, height))
        # Update the cell for this time
        cellUpdate(config, nextConfig, random_numbers, fire_area, fire_data, fire_num)
        # Switch next over to current
        config, nextConfig = nextConfig, config
        # For debugging
        if _ % 10000 == 0:
            print(_)


# Call the updating function
updating(config, nextConfig, fire_area, fire_data, fire_num)

# Filter out 0 area
fire_area = fire_area[fire_area != 0]

# Create buckets to hold the values
num, bucket = histogram(fire_area.astype(np.int32), bins=np.arange(1, width * height, 1000))
# Plot on log log
loglog(bucket[:-1], num / float(Ns))
axes = gca()
axes.set_xlabel("Af")
axes.set_ylabel("Nf/Ns")
title("alpha="+str(dispersion/ignition))
show()
