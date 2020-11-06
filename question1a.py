# Import all the libraries needed for plotting and calculating the transient of the Lorenz system
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Setup the parameters for plotting - currently active is the figure size, label size on the axes, line width,
# and tick size
params = {'figure.figsize': (8, 8),
          'axes.labelsize': 40,
          'lines.linewidth': 2,
          # 'text.fontsize': 12,
          # 'lines.color': 'r',
          'xtick.labelsize': 20,
          'ytick.labelsize': 20,
          # 'legend.fontsize': 10,
          # 'title.fontsize': 12,
          # 'text.usetex': False,
          # 'font': 'Helvetica',
          # 'mathtext.bf': 'helvetica:bold',
          # 'xtick.major.pad': 6,
          # 'ytick.major.pad': 6,
          # 'xtick.major.size': 5,
          # 'ytick.major.size': 5,
          # 'xtick.minor.size': 3,      # minor tick size in points
          # 'xtick.major.width': 1.,    # major tick width in points
          # 'xtick.minor.width': 1.,    # minor tick width in points
          # 'ytick.minor.size': 3,      # minor tick size in points
          # 'ytick.major.width': 1.,    # major tick width in points
          # 'ytick.minor.width': 1.,    # minor tick width in points
          # 'tick.labelsize': 'small'
          }

# Tell matplotlib that we want to use the above object for our settings
plt.rcParams.update(params)


# Define the Lorenz system for a particular time. Input is current x, t and the system parameters.
# Output is a vector of the new x,y,z gradient from the Lorenz equations
def dynamics(x, t, sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    # Initialise the output gradient vector for this x,y,z tuple
    xdot = np.zeros(3)
    # Calculate the gradient in x,y,z
    xdot[0] = sigma * (x[1] - x[0])
    xdot[1] = x[0] * (rho - x[2]) - x[1]
    xdot[2] = x[0] * x[1] - beta * x[2]
    # Return gradient vector
    return xdot


# Create a initial x,y,z from a uniform probability range (0.1,0.5)
x_init = np.random.uniform(0.1, 0.5, 3)

# Initial time of the system (starting from t=0)
t_init = 0;
# Final time that we are calculating to for this system
t_final = 100;
# The length of each step we are taking (0.01s)
t_step = 0.01
# Create the time points of the system
tpoints = np.arange(t_init, t_final, t_step)

# Number of points that are 80% of the total transient response
transient = int(0.8 * len(tpoints))

# Define system parameters being used in the calculation
sigma = 10.0;
beta = 8.0 / 3.0;
rho = 28.0;
alpha = 0.03
# Use ODE integrate to determine the transient response of the system by passing in the dynamics function. The max
# step size in the calculation is 0.01 from hmax
y = odeint(dynamics, x_init, tpoints, args=(sigma, beta, rho), hmax=0.01)

# Start the plot
plt.figure()
# Plot the values. The first column from y is the x-values, second column is the y-values
plt.plot(y[:, 0], y[:, 1], 'k')
# Add LaTeX axis labels of x, y
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
# Set plot limits from -40, 40 in both x and y
plt.xlim([-40, 40])
plt.ylim([-40, 40])
# Use a tight layout for the labels and axes
plt.tight_layout()
