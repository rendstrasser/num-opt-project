import numpy as np
from dataclasses import dataclass, make_dataclass
from typing import Callable, Tuple
from decimal import Decimal

from numpy.lib import gradient

@dataclass
class MinimizationProblem:
    """ 
    Data class containing all necessary information of a minimization problem to support 
    steepest descent, newton, quasi-newton and conjugate minimization.
    """
    f: Callable[[np.ndarray], np.ndarray]
    gradient_f: Callable[[np.ndarray], np.ndarray]
    hessian_f: Callable[[np.ndarray], np.ndarray]
    solution: np.ndarray

@dataclass
class DirectionState:
    """
    Data class containing the state of a direction calculation iteration, 
    holding interesting data for consumers and for the next iteration of the direction calculation.
    """
    x: np.ndarray
    direction: np.ndarray
    gradient: np.ndarray

@dataclass
class BfgsQuasiNewtonState(DirectionState):
    H: np.ndarray

_epsilon = np.sqrt(np.finfo(float).eps)

def find_minimizer(
    problem: MinimizationProblem, 
    x0: np.ndarray, 
    direction_method: Callable[[np.ndarray, MinimizationProblem, DirectionState], DirectionState], 
    a0 = 1, 
    tolerance = 1e-5, 
    max_iter = 10_000):

    x = x0

    direction_state = None
    gradients = []
    for i in range(max_iter):
        x, direction_state = _backtracking_line_search(problem, x, direction_state, direction_method, a0)
        grad_norm = np.linalg.norm(direction_state.gradient)
        gradients.append(grad_norm)

        if grad_norm < tolerance:
            break

    return x, gradients

# calculates conjugate direction with Fletcher-Reeves method
def fr_conjugate_direction(x, problem: MinimizationProblem, prev_state: DirectionState):
    grad = problem.gradient_f(x)
    p = -grad
    
    if prev_state is not None:
        prev_grad = prev_state.gradient
        beta = (grad @ grad) / (prev_grad @ prev_grad)
        p += beta * prev_state.direction

    return DirectionState(x, p, grad)

# calculates quasi-newton direction with BFGS method
def bfgs_quasi_newton_direction(x, problem: MinimizationProblem, prev_state: BfgsQuasiNewtonState):
    I = np.identity(len(x))
    H = I
    grad = problem.gradient_f(x)

    if prev_state is not None:
        s = x - prev_state.x
        y = grad - prev_state.gradient
        rho_denominator = y @ s

        if rho_denominator != 0: # safety condition
            rho = 1/rho_denominator
            H = (I - rho * np.outer(s, y)) @ prev_state.H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

    p = -H @ grad
    return BfgsQuasiNewtonState(x, p, grad, H)

# calculates steepest descent direction
def steepest_descent_direction(x, problem: MinimizationProblem, prev_state: DirectionState):
    grad = problem.gradient_f(x)
    p = -grad

    return DirectionState(x, p, grad)

# calculates newton direction
def newton_direction(x, problem: MinimizationProblem, prev_state: DirectionState):
    hessian = problem.hessian_f(x)
    hessian_inv = np.linalg.inv(hessian)
    grad = problem.gradient_f(x)
    p = -hessian_inv @ grad

    return DirectionState(x, p, grad)

# performs backtracking line search
def _backtracking_line_search(
    problem: MinimizationProblem, 
    x: np.ndarray, 
    prev_p_state: DirectionState,
    direction_method: Callable[[np.ndarray, MinimizationProblem, DirectionState], DirectionState], 
    a0 = 1, c = 0.4, p = 0.8) -> Tuple[np.ndarray, DirectionState]:

    a = a0

    p_state = direction_method(x, problem, prev_p_state)

    # first wolfe condition
    while problem.f(x + a * p_state.direction) > (problem.f(x) + c * a * (p_state.gradient @ p_state.direction)):
        a *= p 

        if a * np.linalg.norm(p_state.direction) < _epsilon:
            # step must not become smaller than precision
            break

    return x + a * p_state.direction, p_state