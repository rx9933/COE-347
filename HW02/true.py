import numpy as np
from scipy.integrate import solve_ivp
from functools import wraps

def exact_solution(f):

    
    sol = solve_ivp(f, [0, 1], [0], method='RK45', 
                    rtol=1e-12, atol=1e-12, dense_output=True)
    return sol

def compute_error(y_numerical, y_exact):
    return np.abs(y_numerical[-1] - y_exact[-1]) #np.linalg.norm(y_numerical - y_exact, ord=2) / np.linalg.norm(y_exact, ord=2)



class FunctionCounter:
    """Decorator class to count function evaluations"""
    def __init__(self, func):
        self.func = func
        self.count = 0
    
    def __call__(self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)
    
    def reset(self):
        self.count = 0
    
    def get_count(self):
        return self.count

def count_evals(func):
    """Decorator to count function evaluations in ODE solvers"""
    @wraps(func)
    def wrapper(dy_dx, y0, x_range, h, return_evals=False, **kwargs):
        # Create a counted version of the function
        counter = FunctionCounter(dy_dx)
        
        # Run the solver with the counted function
        result = func(counter, y0, x_range, h, **kwargs)
        
        if return_evals:
            return (*result, counter.get_count())
        return result
    
    return wrapper