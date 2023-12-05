import numpy as np
import torch
from matplotlib import pyplot as plt

import fossil
from fossil import plotting, control, ActivationType
from fossil.control import ControlAffineControllableDynamicalModel, DynamicalModel
from fossil.plotting import benchmark_3d


class SingleIntegrator(ControlAffineControllableDynamicalModel):
    """
    Single integrator system. X=[x, y], U=[vx, vy]
    dX/dt = [vx, vy]
    """

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
            gx = torch.eye(x.shape[1])[None].repeat((x.shape[0], 1, 1))
        return gx

    def gx_smt(self, x: list) -> np.ndarray | torch.Tensor:
        assert isinstance(x, list), "expected list of symbolic state variables, [x0, x1, ...]"
        return np.eye(len(x))

def generate_pad_data(D: fossil.domains.Set, N: int, M: int, idx: list[int]):
    """
    Generate padded data for a given domain D.
    :param D: domain
    :param N: number of data points
    :param M: number of total dimensions
    :param idx: list of indices of dimensions from domain D
    """
    assert D.dimension == len(idx), "dimension of domain D must match length of idx"
    data = torch.zeros((N, M))
    data[:, idx] = D._generate_data(N)
    return data

def main():
    seed = 0
    system = SingleIntegrator

    np.random.seed(seed)

    XD = fossil.domains.Rectangle(vars=["x0", "x1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
    UD = fossil.domains.Rectangle(vars=["u0", "u1"], lb=(-5.0, -5.0), ub=(5.0, 5.0))
    XI = fossil.domains.Rectangle(vars=["x0", "x1"],lb=(-5.0, -5.0), ub=(-4.0, -4.0))
    XU = fossil.domains.Sphere(vars=["x0", "x1"], centre=[0.0, 0.0], radius=1.0, dim_select=[0, 1])

    sets = {
        fossil.XD: XD,
        fossil.UD: UD,
        fossil.XI: XI,
        fossil.XU: XU,
    }
    data = {
        fossil.XD: lambda: torch.concatenate([XD.generate_data(1000), UD.generate_data(1000)], dim=1),
        fossil.XI: XI._generate_data(400),
        fossil.XU: XU._generate_data(400),
    }

    # define NN parameters
    activations = [fossil.ActivationType.RELU, fossil.ActivationType.LINEAR]
    n_hidden_neurons = [10] * len(activations)

    opts = fossil.CegisConfig(
        N_VARS=2,
        N_CONTROLS=2,
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

    result = fossil.synthesise(
        opts,
    )

    levels = [[0.0]]

    #ax1 = benchmark_plane(model, certificate, domains, levels, xrange, yrange)
    ax2 = benchmark_3d([result.cert], opts.DOMAINS, levels, [-5.0, 5.0], [-5.0, 5.0])
    #ax3 = benchmark_lie(model, certificate, domains, levels, xrange, yrange)

    plt.show()


if __name__ == "__main__":
    main()
