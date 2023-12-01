import unittest

import numpy as np
import torch
import z3

from fossil.control import ControlAffineControllableDynamicalModel


class SingleIntegrator(ControlAffineControllableDynamicalModel):
    @property
    def n_vars(self):
        return 2

    @property
    def n_u(self):
        return 2
    def fx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert len(x.shape) == 3, "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            fx = np.zeros_like(x)
        else:
            fx = torch.zeros_like(x)
        return fx

    def fx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(x, list), "expected list of symbolic state variables, [x0, x1, ...]"
        return np.zeros(len(x))

    def gx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert len(x.shape) == 3, "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            gx = np.eye(x.shape[1])[None].repeat(x.shape[0], axis=0)
        else:
            gx = torch.eye(x.shape[1])[None].repeat(x.shape[0], axis=0)
        return gx

    def gx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(x, list), "expected list of symbolic state variables, [x0, x1, ...]"
        return np.eye(len(x))

class TestControlAffineDynamicalSystem(unittest.TestCase):

    def test_single_integrator(self):

        x = np.zeros((10, 2))
        u = np.ones((10, 2))
        T = 10.0
        dt = 0.1

        f = SingleIntegrator()

        t = dt
        while t < T:
            x = x + dt * f(x, u)
            t += dt

        self.assertTrue(np.allclose(x, 10.0 * np.ones_like(x)), f"got {x}")

    def test_single_integrator_z3(self):
        state_vars = ["x", "y"]
        input_vars = ["vx", "vy"]
        x = [z3.Real(var) for var in state_vars]
        u = [z3.Real(var) for var in input_vars]

        f = SingleIntegrator()

        xdot = f.f(x, u)

        self.assertTrue(str(xdot[0]) == input_vars[0], "expected xdot = vx, got {xdot[0]}")
        self.assertTrue(str(xdot[1]) == input_vars[1], "expected ydot = vy, got {xdot[1]}")





