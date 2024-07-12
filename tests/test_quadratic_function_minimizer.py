import numpy as np
from functions.quadratic_function import QuadraticFunction


def test_quadratic_function_minimizer():
    # Define parameters for the quadratic function
    a = np.array([[3, 2], [2, 6]])
    b = np.array([2, -8])
    # c = np.array([0])
    # x_init = np.array([0, 0])  # Initial guess (not used in closed-form solution)

    # Create an instance of QuadraticFunction
    # quad_func = QuadraticFunction(a, b, c, x_init)
    quad_func = QuadraticFunction(a, b)

    # Compute the closed-form minimizer
    x_min = quad_func.closed_form_minimizer()

    # Compute the gradient at the minimizer
    gradient_at_min = quad_func.gd(x_min)

    # Check if the gradient is close to zero
    if np.allclose(gradient_at_min, np.zeros_like(gradient_at_min)):
        print(f"The closed-form minimizer {x_min} is correct.")
    else:
        print(f"The closed-form minimizer {x_min} is not correct.")
        print(f"Gradient at minimizer: {gradient_at_min}")


if __name__ == "__main__":
    test_quadratic_function_minimizer()