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

# Define the coupling strength matrix used in calcs - contains the identity times against the strength
coupleStrength = np.array([
# x y z
 [0.4,0,0], #x
 [0,0.4,0], #y
 [0,0,0.4]  #x
])

# Generate random starting values
x_init = np.random.uniform(0.1, 0.5, 6)
# Solve the transient response
y = odeint(coupling, x_init, tpoints, args=(coupleStrength,sigma, beta, rho), hmax=0.01)

# Plot the transient response
fig = plt.figure(figsize=(12, 10))
plt.subplot(311)
plt.plot(y[-5000:, 0])
plt.plot(y[-5000:, 3])
plt.ylabel(r'$x(t)$', fontsize=20)
plt.subplot(312)
plt.plot(y[-5000:, 1])
plt.plot(y[-5000:, 4])
plt.ylabel(r'$y(t)$', fontsize=20)
plt.subplot(313)
plt.plot(y[-5000:, 2])
plt.plot(y[-5000:, 5])
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$z(t)$', fontsize=20)
plt.show()

# Define the steps and data containers for the E plot
minCouple = 0.40
maxCouple = 0.505
stepCouple = 0.05
stepNumCouple = int(np.ceil((maxCouple-minCouple)/stepCouple))
stepsCouple = np.arange(minCouple,maxCouple,stepCouple)
dataCouple = np.zeros(stepNumCouple)

# How many times we should average in the n plots (higher is better but slower)
n = 20

# Iterate over n times for average
for i in range(n):
    # Generate a random start
    x_init = np.random.uniform(0.1, 0.5, 6)
    # Enumerate over the number of steps we need in the alpha of the E diagram
    for i,x in enumerate(stepsCouple):
        # Create the identity coupling matrix multiplied by alpha
        coupleStrength = np.diag([x, x, x])
        # Run odeint to get the transient of this coupling and initial starting values
        y = odeint(coupling, x_init, tpoints, args=(coupleStrength, sigma, beta, rho), hmax=0.01)
        # Perform the integration of the norm between the three axes. Assuming L2 space
        dataCouple[i] += simps(np.sqrt((y[:, 0] - y[:, 3])**2 +
                                (y[:, 1] - y[:, 4])**2 +
                                (y[:, 2] - y[:, 5])**2
                                ), tpoints)

# Plotting code
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'E')
ax.plot(stepsCouple, ((dataCouple/n) / t_final))
plt.show()