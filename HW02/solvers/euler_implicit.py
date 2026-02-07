import numpy as np
from scipy.optimize import fsolve
from true import *

@count_evals
def implicit_euler(dy_dx, y0, x_range, h):
    x0, x_end = x_range
    n_steps = int((x_end - x0) / h) + 1
    x = np.linspace(x0, x_end, n_steps)
    y = np.zeros(n_steps)
    y[0] = y0
    
    for i in range(n_steps - 1):
        # Define function for Newton iteration
        def F(y_next):
            return y_next - y[i] - h * dy_dx(x[i+1], y_next)

        # Initial guess (explicit Euler)
        y_guess = y[i] + h * dy_dx(x[i], y[i])
        y[i+1] = fsolve(F, y_guess)[0]
    
    return x, y