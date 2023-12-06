import random

import numpy as np
import torch
from matplotlib import pyplot as plt

import fossil
from fossil import plotting, control, ActivationType
from fossil.control import ControlAffineControllableDynamicalModel, DynamicalModel, GeneralController
from fossil.plotting import benchmark_3d, benchmark_plane, benchmark_lie


class SingleIntegrator(ControlAffineControllableDynamicalModel):
    """
    Single integrator system. X=[x, y], U=[vx, vy]
    dX/dt = [vx, vy]
    """

    @property
    def n_vars(self) -> int:
        return 2

    @property
    def n_u(self) -> int:
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
            gx = torch.eye(x.shape[1])[None].repeat((x.shape[0], 1, 1))
        return gx

    def gx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(x, list), "expected list of symbolic state variables, [x0, x1, ...]"
        return np.eye(len(x))


class DoubleIntegrator(ControlAffineControllableDynamicalModel):
    """
    Single integrator system. X=[x, y], U=[vx, vy]
    dX/dt = [vx, vy]
    """

    @property
    def n_vars(self) -> int:
        return 4

    @property
    def n_u(self) -> int:
        return 2

    def fx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert len(x.shape) == 3, "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            vx, vy = x[:, 2, :], x[:, 3, :]
            fx = np.concatenate([vx, vy, np.zeros_like(vx), np.zeros_like(vy)], axis=1)
            fx = fx[:, :, None]
        else:
            vx, vy = x[:, 2, :], x[:, 3, :]
            fx = torch.cat([vx, vy, torch.zeros_like(vx), torch.zeros_like(vy)], dim=1)
            fx = fx[:, :, None]
        return fx

    def fx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(x, list), "expected list of symbolic state variables, [x0, x1, ...]"
        vx, vy = x[2], x[3]
        return np.array([vx, vy, 0, 0])

    def gx_torch(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        assert len(x.shape) == 3, "expected batched input with shape (batch_size, state_dim, 1)"
        if isinstance(x, np.ndarray):
            gx = np.zeros((self.n_vars, self.n_u))
            gx[2:, :] = np.eye(self.n_u)
            gx = gx[None].repeat(x.shape[0], axis=0)
        else:
            gx = torch.zeros((self.n_vars, self.n_u))
            gx[2:, :] = torch.eye(self.n_u)
            gx = gx[None].repeat((x.shape[0], 1, 1))
        return gx

    def gx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(x, list), "expected list of symbolic state variables, [x0, x1, ...]"
        gx = np.zeros((self.n_vars, self.n_u))
        gx[2:, :] = np.eye(self.n_u)
        return gx


def main():
    seed = 42
    system_name = "single_integrator"

    if system_name == "single_integrator":
        system = SingleIntegrator
        XD = fossil.domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        UD = fossil.domains.Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        XI = fossil.domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(-4.0, -4.0))
        XU = fossil.domains.Sphere(vars=["x0", "x1"], centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1])
    elif system_name == "double_integrator":
        system = DoubleIntegrator
        XD = fossil.domains.Rectangle(vars=["x0", "x1", "x2", "x3"],
                                      lb=(-5.0, -5.0, -5.0, -5.0),
                                      ub=(5.0, 5.0, 5.0, 5.0))
        UD = fossil.domains.Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
        XI = fossil.domains.Rectangle(vars=["x0", "x1", "x2", "x3"],
                                      lb=(-5.0, -5.0, -5.0, -5.0),
                                      ub=(-4.0, -4.0, 5.0, 5.0))
        XU = fossil.domains.Rectangle(vars=["x0", "x1", "x2", "x3"],
                                      lb=(-1.0, -1.0, -5.0, -5.0),
                                      ub=(1.0, 1.0, 5.0, 5.0))
    else:
        raise NotImplementedError(f"System {system_name} not implemented")

    # seeding
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)



    sets = {
        fossil.XD: XD,
        fossil.UD: UD,
        fossil.XI: XI,
        fossil.XU: XU,
    }
    data = {
        fossil.XD: lambda: torch.concatenate([XD.generate_data(400), UD.generate_data(400)], dim=1),
        fossil.XI: XI._generate_data(400),
        fossil.XU: XU._generate_data(400),
    }

    # define NN parameters
    activations = [fossil.ActivationType.RELU, fossil.ActivationType.LINEAR]
    n_hidden_neurons = [10] * len(activations)

    opts = fossil.CegisConfig(
        N_VARS=system().n_vars,
        N_CONTROLS=system().n_u,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=fossil.CertificateType.CBF,
        TIME_DOMAIN=fossil.TimeDomain.CONTINUOUS,
        VERIFIER=fossil.VerifierType.Z3,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=False,
        CEGIS_MAX_ITERS=25,
        VERBOSE=1,
        SEED=167,
    )

    levels = [[0.0]]

    result = fossil.synthesise(
        opts,
    )

    ctrl = control.DummyController(
        inputs=opts.N_VARS,
        output=opts.N_CONTROLS,
        const_out=1.0
    )
    closed_loop_model = control.GeneralClosedLoopModel(result.f, ctrl)

    if XD.dimension == 2:
        xrange = (XD.lower_bounds[0], XD.upper_bounds[0])
        yrange = (XD.lower_bounds[1], XD.upper_bounds[1])

        ax1 = benchmark_plane(closed_loop_model, [result.cert], opts.DOMAINS, levels, xrange, yrange)
        ax2 = benchmark_3d([result.cert], opts.DOMAINS, levels, xrange, yrange)
        ax3 = benchmark_lie(closed_loop_model, [result.cert], opts.DOMAINS, levels, xrange, yrange)

        plt.show()


if __name__ == "__main__":
    main()
