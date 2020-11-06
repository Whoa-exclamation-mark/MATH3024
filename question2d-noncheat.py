# Import relevant libraries
import networkx as nx
from pylab import *

# State numbers for each node
R, P, S = 1, 2, 3

# Define an array of which we can lookup who wins in a gam
game = np.array([
    [R, P, R],
    [P, P, S],
    [R, S, S]
])

# Number of nodes
N = 50

# Define number of samples in steady state response
samples = 50

# Name to display on charts
name = "Barabasi"

# What values to sweep over
# prop_values = [0.05, 0.1, 0.2, 0.3]
prop_values = [3, 5, 7, 11]

# What graph function to use
# graph_function = lambda n, prop: nx.binomial_graph(n, prop)
# graph_function = lambda n, prop: nx.newman_watts_strogatz_graph(n, 3, prop)
graph_function = lambda n, prop: nx.barabasi_albert_graph(n, prop)


# Updating function
def update(nodes, neighbors):
    # Choose a random player from nodes
    player = choice(list(nodes))
    # Find the neighbors of the node
    nbs = list(neighbors(player))

    # Does this node have neighbors?
    if nbs:
        # Get a challenger node from all the nodes connected to player
        challenger = choice(nbs)
        # Run the game by looking up from state table
        result = game[nodes[player]['state'] - 1, nodes[challenger]['state'] - 1]
        # Set the wining result state to both the player and the challenger
        nodes[player]['state'] = result
        nodes[challenger]['state'] = result

    # Get all the states in a format of {Rock: # of rock, Paper: # of paper...}
    un, counts = unique(array(
        [nodes[i]["state"] for i in nodes]
    ), return_counts=True)
    zipped = dict(zip(un, counts))
    # Create an array to append the result of the states to
    appeding = []
    # Check if Rock is in the zipped dictionary
    if 1 in zipped:
        # If so append the value of rock at index of 0
        appeding.append(zipped[1])
    else:
        # Else append 0 to indicate there is nothing
        appeding.append(0)
    # Do the same for Paper and Scissors
    if 2 in zipped:
        appeding.append(zipped[2])
    else:
        appeding.append(0)
    if 3 in zipped:
        appeding.append(zipped[3])
    else:
        appeding.append(0)

    # Return final array of data in [# rock, # paper, # scissors]
    return appeding


# Do one transient run to show what the transient looks like
# Setup an array to hold the transient data
time_data = []
# Generate a random graph of the first property
g = graph_function(N, prop_values[1])
# Go through and set states
for i in g.nodes:
    g.nodes[i]['state'] = choice(range(1, 4))

time = 0
# Go through the transient until reach steady state (one reaches 300 -> nothing can happen after this)
while True:
    # Get the data from the update function
    data = update(g.nodes, g.neighbors)
    # Append to time data array
    time_data.append(data)
    # Check if we are at steady state by taking the max of the updated data
    time += 1
    if max(data) == N or time == 200000:
        # If we are break out of loop
        break

# Plot the transient response
subplot(1,1,1)
plot(array(time_data))
title(name)
ylabel("# of type")
xlabel("t")
show()

# Now calculate the steady state time for a sweep over prop_values
# Setup data holding array of length of prop_values by # of samples (20)
time = np.zeros((len(prop_values), samples))
# Iterate over prop_values for sweep
for index, j in enumerate(prop_values):
    # Iterate samples times for sampling
    for _ in range(samples):
        # Set current time to 0 (new run) -> hold the time it takes to reach steady state
        curr_time = 0
        # Create random graph with j as the property
        g = graph_function(N, j)
        # Setup random states for all nodes
        for i in g.nodes:
            g.nodes[i]['state'] = choice(range(1, 4))
        # Go through the transient until reach steady state (one reaches 300 -> nothing can happen after this)
        while True:
            # Get the data from the update function
            data = update(g.nodes, g.neighbors)
            # Check if we reached steady state of one of state or time is 200000 (we aren't reaching steady state fast)
            if max(data) == N or curr_time == 200000:
                # If we have exit loop
                break
            # Else increment current time by one
            curr_time += 1
        # Debugging - printing out the current time value reached
        print(curr_time)
        # Add the found steady state time to time array
        time[index, _] = curr_time

# Setup plot
ax = subplot(1, 1, 1)
# Use boxplot as we are sampling random data (shows variance as we used a low # of samples)
ax.boxplot(time.T)
# Set labels and titles
xlabel("p/m")
ylabel("t")
title(name)
# Set xticks to be the prop_values
xticks(range(1, 5), prop_values)
# Tell pyplot to plot from stack
show()


