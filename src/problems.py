import numpy as np
from scipy.stats import ortho_group

from src.algorithms import MinimizationProblem

# original function
def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2

# first derivative
def d_rosenbrock(x):
    dx = 400*x[0]**3-400*x[0]*x[1]+2*x[0]-2
    dy = 200*x[1]-200*x[0]**2

    return np.array([dx, dy])

# second derivative
def d2_rosenbrock(x):
    ddx = 1200*x[0]**2-400*x[1]+2
    ddxy = -400*x[0]
    ddy = 200

    return np.array([[ddx, ddxy], [ddxy, ddy]])

ROSENBROCK_PROBLEM = MinimizationProblem(rosenbrock, d_rosenbrock, d2_rosenbrock, np.array([1, 1]))

def create_quadratic_problem(n, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    A_generator = np.random.randint(low=1, high=10, size=(n, n))

    A = A_generator @ A_generator.T # -> positive definite matrix
    solution = np.random.randint(low=1, high=10, size=(n))
    b = A @ solution

    def f(x):
        return 1/2 * x @ A @ x - b @ x
    
    def d_f(x):
        return A @ x - b
    
    def d2_f(x):
        return A

    return MinimizationProblem(f, d_f, d2_f, solution)

def create_non_quadratic_problem(random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    a = np.random.randint(low=-10, high=-1)
    b = np.random.randint(low=-1, high=7)
    c = np.random.randint(low=7, high=15)

    # 2 minimizers possible
    solution = [a, c]

    # integral of (x-a)(x-b)(x-c)
    def f(x):
        return 1/12 * x[0]*(-4*x[0]**2* (a+b+c) + 6*x[0]*(a*(b+c) + b*c) - 12*a*b*c + 3*x[0]**3)
    
    def d_f(x):
        return (x-a)*(x-b)*(x-c)
    
    def d2_f(x):
        return np.array([a*(b+c-2*x) + b*(c-2*x) + x*(3*x-2*c)])

    return MinimizationProblem(f, d_f, d2_f, solution)