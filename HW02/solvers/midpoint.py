import numpy as np
from scipy.optimize import fsolve
from true import *

@count_evals
def midpoint(dy_dx, y0, x_range, h):

    x0, x_end = x_range
    n_steps = int((x_end - x0) / h) + 1
    x = np.linspace(x0, x_end, n_steps)
    y = np.zeros(n_steps)
    y[0] = y0
    
    for i in range(n_steps - 1):
        k1 = dy_dx(x[i], y[i])
        k2 = dy_dx(x[i] + h/2, y[i] + h/2 * k1)
        y[i+1] = y[i] + h * k2
    
    return x, y

@count_evals
def implicit_midpoint(dy_dx, y0, x_range, h):

    x0, x_end = x_range
    n_steps = int((x_end - x0) / h) + 1
    x = np.linspace(x0, x_end, n_steps)
    y = np.zeros(n_steps)
    y[0] = y0

    for i in range(n_steps - 1):
        x_mid = x[i] + h/2  # Midpoint in x
        
        # Define function for implicit equation
        # y_{n+1} = y_n + h * f(x_mid, (y_n + y_{n+1})/2)
        def F(y_next):
            return y_next - y[i] - h * dy_dx(x_mid, (y[i] + y_next)/2)
        
        # Use Newton's method to solve implicit equation
        # Initial guess: explicit midpoint method
        k1 = dy_dx(x[i], y[i])
        y_guess = y[i] + h * dy_dx(x[i] + h/2, y[i] + h/2 * k1)
        
        # Solve using fsolve
        y[i+1] = fsolve(F, y_guess, xtol=1e-5, maxfev=100)[0]
    
    return x, y