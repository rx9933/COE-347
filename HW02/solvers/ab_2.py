import numpy as np
from true import *

@count_evals
def adams_bashforth2(f, y0, x_range, h):

    x0, x_end = x_range
    n_steps = int((x_end - x0) / h) + 1
    x = np.linspace(x0, x_end, n_steps)
    y = np.zeros(n_steps)
    y[0] = y0
    
    # First step using explicit Euler
    y[1] = y[0] + h * f(x[0], y[0])
    
    # Adams-Bashforth 2-step formula
    for i in range(1, n_steps - 1):
        y[i+1] = y[i] + h/2 * (3*f(x[i], y[i]) - f(x[i-1], y[i-1]))
    
    return x, y