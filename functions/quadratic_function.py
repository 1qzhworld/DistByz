# functions/quadratic_function.py

import numpy as np


class QuadraticFunction:
    def __init__(self, a: np.ndarray, b: np.ndarray, c: np.ndarray, x_init: np.ndarray):
        """
        x^T @ a @ x + b @ x + c
        :param a: square matrix
        :param b: vector
        :param c: bias
        :param x_init: variable
        """
        self.a = a
        self.b = b
        self.c = c
        self.x_init = x_init

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate the quadratic function at point x.
        :param x: np.ndarray, the point at which to evaluate the function.
        :return: float, the function value at x.
        """
        return 0.5 * x.T @ self.a @ x + self.b.T @ x + self.c

    def gd(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the gradient of the quadratic function at point x.
        :param x: np.ndarray, the point at which to compute the gradient.
        :return: np.ndarray, the gradient at x.
        """
        return 0.5 * (self.a.T + self.a) @ x + self.b.flatten()

    def closed_form_minimizer(self) -> np.ndarray:
        """
        Compute the closed-form minimizer of the quadratic function.
        :return: np.ndarray, the minimizer.
        """
        return (-np.linalg.inv(0.5 * (self.a.T + self.a)) @ self.b).flatten()
