import numpy as np
from scipy.integrate import solve_ivp

def exact_solution():
    def ode_func(x, y):
        return -50 * (y - np.cos(x))
    
    sol = solve_ivp(ode_func, [0, 1], [0], method='RK45', 
                    rtol=1e-12, atol=1e-12, dense_output=True)
    return sol

def compute_error(y_numerical, y_exact):
    return np.linalg.norm(y_numerical - y_exact, ord=2) / np.linalg.norm(y_exact, ord=2)

#np.abs(y_numerical[-1] - y_exact[-1])