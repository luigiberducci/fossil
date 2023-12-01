import numpy as np
import torch

import fossil
from fossil import plotting, control, ActivationType
from fossil.control import ControlAffineControllableDynamicalModel


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


def main():
    open_loop = SingleIntegrator
    system = control.GeneralClosedLoopModel.prepare_from_open(open_loop())

    XD = fossil.domains.Rectangle([-5, -5], [5, 5])
    UD = fossil.domains.Rectangle([-5, -5], [5, 5])
    XI = fossil.domains.Rectangle([-5, -5], [-4, -4])
    XU = fossil.domains.Sphere([0.0, 0.0], 1.0)

    sets = {
        fossil.XD: XD,
        fossil.UD: UD,
        fossil.XI: XI,
        fossil.XU: XU,
    }
    data = {
        fossil.XD: XD._generate_data(1000),
        fossil.UD: UD._generate_data(400),
        fossil.XI: XI._generate_data(400),
        fossil.XU: XU._generate_data(400),
    }

    # define NN parameters
    activations = [fossil.ActivationType.RELU, fossil.ActivationType.LINEAR]
    n_hidden_neurons = [10] * len(activations)

    opts = fossil.CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=fossil.CertificateType.CBF,
        TIME_DOMAIN=fossil.TimeDomain.CONTINUOUS,
        VERIFIER=fossil.VerifierType.Z3,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CTRLAYER=[5, 2],
        CTRLACTIVATION=[ActivationType.LINEAR],
        SYMMETRIC_BELT=False,
        CEGIS_MAX_ITERS=25,
        VERBOSE=1,
        SEED=167,
    )

    result = fossil.synthesise(
        opts,
    )
    D = opts.DOMAINS.pop(fossil.XD)
    plotting.benchmark(
        result.f, result.cert, domains=opts.DOMAINS, xrange=[-3, 2.5], yrange=[-2, 1]
    )


if __name__ == "__main__":
    main()
