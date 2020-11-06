# Import relevant libraries
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# What graph should we use + positioning function
# graph_function = lambda n, prop: nx.binomial_graph(n, prop)
graph_function = lambda n, prop: nx.newman_watts_strogatz_graph(n, 3, prop)
# graph_function = lambda n, prop: nx.barabasi_albert_graph(n, prop)
position_function = lambda g: nx.circular_layout(g)
# position_function = lambda g: nx.spectral_layout(g)

# Array of props to sweep through
prop_values = [0.05, 0.1, 0.2, 0.3]
# prop_values = [3, 5, 7, 11]
# How many times should we iterate through each prop value to take an average (reduction of noise)
iter_ave = 5


# Helper function to show graphs
def show_graph(graph):
    # Set positioning
    graph.pos = position_function(graph)
    # Draw the graph
    nx.draw(graph, pos=graph.pos)
    # Tell pyplot to plot the last graph on the stack
    plt.show()


# Generate a graph of small nodes with first prop to compare
graph = graph_function(20, prop_values[0])
# Use helper function to show graph
show_graph(graph)

# Generate a graph of small nodes with last prop to compare
graph = graph_function(20, prop_values[-1])
# Use helper function to show graph
show_graph(graph)

# Dictionaries to hold the functions for the centralities to explore
centralities = {"Degree": nx.degree_centrality,
                "Betweenness": nx.betweenness_centrality,
                # Using lambda function to specify the max iter (Newmann runs into problems)
                "Eigen": lambda g: nx.eigenvector_centrality(g, max_iter=1000),
                "Closeness": nx.closeness_centrality}

# Create plot
fig = plt.figure(figsize=(12, 10))

for name, func in centralities.items():
    # Create empty array for storing the data calculated
    cent_data = np.zeros(len(prop_values))
    # Iterate over the prop values
    for i in range(len(prop_values)):
        # Run iter_ave times
        for _ in range(iter_ave):
            # Create graph with the prop value
            graph = graph_function(300, prop_values[i])
            # find the centrality and average over all the nodes
            cent_data[i] += sum(func(graph).values()) / 300.0
        # Average the calc from above to reduce noise
        cent_data[i] /= iter_ave
    # Get position of the current name to figure out what the index number is for the plotting
    index = list(centralities).index(name)
    # Create Subplot
    ax = fig.add_subplot(2, 2, index + 1)
    # Add name as title to subplot
    ax.title.set_text(name)
    # Plot the property values against centrality data
    ax.plot(prop_values, cent_data)

# Show the plot
plt.show()
