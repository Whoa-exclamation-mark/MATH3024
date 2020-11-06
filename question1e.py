# Import all the libraries needed for plotting and calculating the transient of the Lorenz system
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint, simps
from numba import jit

# Enum of the possible drives
x_couple, y_couple, z_couple = 1,2,3

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
def coupling(x, t, couple_type, sigma=10.0, beta=8.0 / 3.0, rho=28.0):
    # Unpack the x vector into the two systems
    x0, x1, x2, u0, u1, u2 = x
    # Repack the two system states into two differing vectors
    x = np.array([x0, x1, x2])
    u = np.array([u0, u1, u2])
    # First system gradient vector
    xdot = np.zeros(3)
    # Lorenz system of primary system
    xdot[0] = sigma * (x1 - x0)
    xdot[1] = x0 * (rho - x2) - x1
    xdot[2] = x0 * x1 - beta * x2

    # Similar to above but for the second system. Coupling based on couple_type
    udot = np.zeros(3)
    # Check if x coupling
    if couple_type == x_couple:
        # If so set x->x for both systems
        udot[0] = xdot[0]
    else:
        # Else have a separate equation from first system
        udot[0] = sigma * (u1 - u0)
    # Do this for the other 2 systems
    if couple_type == y_couple:
        udot[1] = xdot[1]
    else:
        udot[1] = u0 * (rho - u2) - u1
    if couple_type == z_couple:
        udot[2] = xdot[2]
    else:
        udot[2] = u0 * u1 - beta * u2
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

# Generate random starting values
x_init = np.random.uniform(0.1, 0.5, 6)
# Solve the transient response
y = odeint(coupling, x_init, tpoints, args=(z_couple,sigma, beta, rho), hmax=0.01)

# Plot the transient response
fig = plt.figure(figsize=(12, 10))
plt.subplot(311)
plt.plot(y[-5000:, 0]-y[-5000:, 3])
plt.ylabel(r'$x(t)$', fontsize=20)
plt.subplot(312)
plt.plot(y[-5000:, 1]-y[-5000:, 4])
plt.ylabel(r'$y(t)$', fontsize=20)
plt.subplot(313)
plt.plot(y[-5000:, 2]-y[-5000:, 5])
plt.xlabel(r'$t$', fontsize=20)
plt.ylabel(r'$z(t)$', fontsize=20)
plt.show()