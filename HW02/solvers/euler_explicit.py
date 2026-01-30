import numpy as np

def explicit_euler(dy_dx, y0, x_range, h):
    x0, x_end = x_range
    n_steps = int((x_end - x0) / h) + 1
    x = np.linspace(x0, x_end, n_steps)
    y = np.zeros(n_steps)
    y[0] = y0
    
    for i in range(n_steps - 1):
        y[i+1] = y[i] + h * dy_dx(x[i], y[i])
    
    return x, y