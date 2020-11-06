# Import all the libraries needed for plotting and calculating the transient of the Lorenz system
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, simps
from numba import jit

# Setup the parameters for plotting - currently active is the figure size, label size on the axes, line width,
# and tick size
params = {'figure.figsize': (8, 8),
          'axes.labelsize': 40,
          'lines.linewidth': 2,
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          }

# Tell matplotlib that we want to use the above object for our settings
plt.rcParams.update(params)


# Define the Lorenz system for a particular time. Input is current x, t, coupling strength, the system parameters.
# Output is a vector of the new x,y,z gradient from the Lorenz equations
# Use numba to do loop lifting to speed things up
@jit
def coupling(x, t, coupleStrength ,sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    # Unpack the x vector into the two systems
    x0, x1, x2, u0, u1, u2 = x
    # Repack the two system states into two differing vectors
    x = np.array([x0, x1, x2])
    u = np.array([u0, u1, u2])
    # First system gradient vector
    xdot = np.zeros(3)
    # Lorenz system with coupling. The coupling takes in a matrix and gets a column slice then dots it with the diff
    # between the two system states
    xdot[0] = sigma * (x1 - x0) + np.dot(coupleStrength[:, 0], (u - x))
    xdot[1] = x0 * (rho - x2) - x1 + np.dot(coupleStrength[:, 1], (u - x))
    xdot[2] = x0 * x1 - beta * x2 + np.dot(coupleStrength[:, 2], (u - x))

    # Similar to above but for the second system. Coupling is backward for the second system
    udot = np.zeros(3)
    udot[0] = sigma * (u1 - u0) + np.dot(coupleStrength[:, 0], (x - u))
    udot[1] = u0 * (rho - u2) - u1 + np.dot(coupleStrength[:, 1], (x - u))
    udot[2] = u0 * u1 - beta * u2 + np.dot(coupleStrength[:, 2], (x - u))
    # Combine the two gradient vectors to return back to odeint
    return np.concatenate((xdot, udot))

# Define starting values and ending values of the transient response with timesteps
t_init = 0
t_final = 500
t_step = 0.01
tpoints = np.arange(t_init, t_final, t_step)

# Define system parameters
sigma = 10.0
beta = 8.0 / 3.0
rho = 28.0
alpha = 0.03


# Define a function to generate the E plots for different coupling
def error_graph(start, end, n, coupleMatrix, ax=None):
    # Define the steps and data containers for the E plot
    stepCouple = (end - start) / n
    stepsCouple = np.arange(start, end, stepCouple)
    dataCouple = np.zeros(n)

    # Define the average used in calculations of the plots
    avg = 5

    # Iterate over the number of averages
    for i in range(avg):
        # Generate a random vector for starting values
        x_init = np.random.uniform(0.1, 0.5, 6)
        for i, x in enumerate(stepsCouple):
            # Create the coupling matrix multiplied by alpha
            coupleStrength = coupleMatrix * x
            # Run odeint to get the transient of this coupling and initial starting values
            y = odeint(coupling, x_init, tpoints, args=(coupleStrength, sigma, beta, rho), hmax=0.01)
            # Perform the integration of the norm between the three axes. Assuming L2 space
            dataCouple[i] += simps(np.sqrt((y[:, 0] - y[:, 3]) ** 2 +
                                           (y[:, 1] - y[:, 4]) ** 2 +
                                           (y[:, 2] - y[:, 5]) ** 2
                                           ), tpoints)
    # See if ax was passed through so we can plot if not then generate axis
    ax = ax or plt.gca()
    # Plot with the averaged and 1/T
    ax.plot(stepsCouple, ((dataCouple / avg) / t_final))


# Define a no_sync range (if the system doesn't sync with each other)
no_sync = (10, 15)
# Alpha ranges for the E plots (calculated by manual inspection from before)
loadings = np.array(
    [[(3.5, 4.5), (3, 4), no_sync],
     [(1.5, 2.5), (1, 2), no_sync],
     [no_sync, no_sync, (0.5, 1.5)]])

# Define the axis labels to place on graph
axis_labels = ["x", "y", "z"]

# Set size of axis
plt.rc('axes', labelsize=20)

# Create a null matrix to copy from
null = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
# Create figure
fig = plt.figure(figsize=(12, 10))
# Iterate over the x, y, z twice to create pairings
for x in range(3):
    for y in range(3):
        # Get the alpha ranges
        start, end = loadings[x, y]
        # Copy the null matrix to not effect it
        curr_idenity = np.copy(null)
        # Set the coupling pair of in the coupling matrix
        curr_idenity[x, y] = 1
        # Set the subplot in 3x3 grid
        ax = fig.add_subplot(3, 3, x * 3 + y + 1)
        # Set side labels for x, y, z based on if y or x is 0
        if y == 0:
            ax.set_ylabel(r'\textbf{' + axis_labels[x] + '}')
        if x == 0:
            ax.xaxis.set_label_position('top')
            ax.set_xlabel(r'\textbf{' + axis_labels[y] + '}')
        # Run error graph function with the current pairing
        error_graph(start, end, 10, curr_idenity, ax)

# Add overall labels
fig.text(0.5, 0.04, r'$\alpha$', ha='center', fontsize=24)
fig.text(0.04, 0.5, r'E', va='center', rotation='vertical', fontsize=24)
# Show plot
plt.show()
