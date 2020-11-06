# Import all the libraries needed for plotting and calculating the transient of the Lorenz system
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, simps
from numba import jit

# Define the Lorenz system for a particular time. Input is current x, t, coupling strength, noise, and the system parameters.
# Output is a vector of the new x,y,z gradient from the Lorenz equations
# Use numba to do loop lifting to speed things up
@jit
def coupling_noise(x, coupleStrength, D, sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    # Create a lambda for the zeta term (so each time it produces a differing random value)
    zeta = lambda: np.sqrt(D) * np.random.normal(loc=0.0, scale=1)

    # Unpack the x vector into the two systems
    x0, x1, x2, u0, u1, u2 = x
    # Repack the two system states into two differing vectors
    x = np.array([x0, x1, x2])
    u = np.array([u0, u1, u2])
    # First system gradient vector
    xdot = np.zeros(3)
    # Lorenz system with coupling. The coupling takes in a matrix and gets a column slice then dots it with the diff
    # between the two system states
    xdot[0] = sigma * (x1 - x0) + np.dot(coupleStrength[:, 0], (u - x)) + zeta()
    xdot[1] = x0 * (rho - x2) - x1 + np.dot(coupleStrength[:, 1], (u - x)) + zeta()
    xdot[2] = x0 * x1 - beta * x2 + np.dot(coupleStrength[:, 2], (u - x)) + zeta()

    # Similar to above but for the second system. Coupling is backward for the second system
    udot = np.zeros(3)
    udot[0] = sigma * (u1 - u0) + np.dot(coupleStrength[:, 0], (x - u)) + zeta()
    udot[1] = u0 * (rho - u2) - u1 + np.dot(coupleStrength[:, 1], (x - u)) + zeta()
    udot[2] = u0 * u1 - beta * u2 + np.dot(coupleStrength[:, 2], (x - u)) + zeta()

    # Combine the two gradient vectors to return back
    return np.concatenate((xdot, udot))


# Define starting values and ending values of the transient response with timesteps
t_init = 0
t_end = 100
N = 10000
dt = float(t_end - t_init) / N

# Define the starting points
x_init = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

# Create the array for the time
ts = np.arange(t_init, t_end + dt, dt)
# Create the array to store the transient response
xs = np.zeros((N + 1, 6))

# Start at x_init
xs[0] = x_init

# Set the coupling strength with the identity coupling
coupleStrength = np.identity(3) * 0.7

# Iterate over the full response
for i in range(1, ts.size):
    # Get the last x value
    x = xs[i - 1]
    # Add the last x value with the result of coupling_noise multiplied by the change in time
    xs[i] = x + coupling_noise(x, coupleStrength, 0.2) * dt

# Plot the transient of the system
fig = plt.figure(figsize=(12, 10))
plt.subplot(311)
plt.plot(xs[:, 0])
plt.plot(xs[:, 3])
plt.ylabel(r'$x(t)$', fontsize=20)
plt.subplot(312)
plt.plot(xs[:, 1])
plt.plot(xs[:, 4])
plt.ylabel(r'$y(t)$', fontsize=20)
plt.subplot(313)
plt.plot(xs[:, 2])
plt.plot(xs[:, 5])
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$z(t)$', fontsize=20)
plt.show()

# Define the average we are using (as system is noisy and chaotic)
avg = 50

# Define our solver function as loop lifting is needed to solve the system quickly
@jit
def solver(t_init, t_end, x_init, N, D, coupleStrength):
    # Calcuate the change in time
    dt = float(t_end - t_init) / N
    # Create time array
    ts = np.arange(t_init, t_end + dt, dt)
    # Create the array to store the transient response
    xs_blank = np.zeros((N + 1, 6))
    # Start at x_init
    xs_blank[0] = x_init
    # Copy to not effect the blank
    xs = np.copy(xs_blank)
    # Iterate over the full response
    for j in range(1, ts.size):
        # Get the last x value
        x = xs[j - 1]
        # Add the last x value with the result of coupling_noise multiplied by the change in time
        xs[j] = x + coupling_noise(x, coupleStrength, D) * dt
    return xs


# Define helper function for graphing
def error_graph_noise(start, end, n, D, ax=None):
    # Define constants and containers needed for the E plots
    stepCouple = (end - start) / n
    stepsCouple = np.arange(start, end, stepCouple)
    dataCouple = np.zeros(n)

    # Iterate over average the E plots
    for i in range(avg):
        # Enumerate over the number of steps we need in the alpha of the E diagram
        for i, x in enumerate(stepsCouple):
            # Define the coupling matrix by alpha times identity coupling
            coupleStrength = np.identity(3) * x
            # Define starting values and ending values of the transient response with timesteps
            t_init = 0
            t_end = 100
            N = 10000  # Compute 1000 grid points

            # Define initial random values from 0.1 to 0.5 (6 values for two systems 3d)
            x_init = np.arange(0.1,0.5,6)

            # Solve the system with function above
            xs = solver(t_init, t_end, x_init, N, D, coupleStrength)

            # Perform the integration of the norm between the three axes. Assuming L2 space
            dataCouple[i] += simps(np.sqrt((xs[:, 0] - xs[:, 3]) ** 2 +
                                           (xs[:, 1] - xs[:, 4]) ** 2 +
                                           (xs[:, 2] - xs[:, 5]) ** 2
                                           ), ts)
    # See if ax was passed through so we can plot if not then generate axis
    ax = ax or plt.gca()
    # Plot with the averaged and 1/T
    ax.plot(stepsCouple, ((dataCouple / avg) / (t_end)))


# Plotting code for four graphs
fig = plt.figure(figsize=(12, 10))
plt.rc('axes', labelsize=20)

ax = fig.add_subplot(2, 2, 1)
ax.xaxis.set_label_position('top')
ax.set_xlabel(r'D=0.05')
ax.set_ylabel("")
error_graph_noise(0.5, 1.5, 5, 0.05, ax)

ax = fig.add_subplot(2, 2, 2)
ax.xaxis.set_label_position('top')
ax.set_xlabel(r'D=0.1')
ax.set_ylabel("")
error_graph_noise(0.5, 1.5, 5, 0.1, ax)

ax = fig.add_subplot(2, 2, 3)
ax.xaxis.set_label_position('top')
ax.set_xlabel(r'D=0.15')
ax.set_ylabel("")
error_graph_noise(0.5, 1.5, 5, 0.15, ax)

ax = fig.add_subplot(2, 2, 4)
ax.xaxis.set_label_position('top')
ax.set_xlabel(r'D=0.2')
ax.set_ylabel("")
error_graph_noise(0.5, 1.5, 5, 0.2, ax)

fig.text(0.5, 0.04, r'$\alpha$', ha='center', fontsize=24)
fig.text(0.04, 0.5, r'E', va='center', rotation='vertical', fontsize=24)
plt.show()
