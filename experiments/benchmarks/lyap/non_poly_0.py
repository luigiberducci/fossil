# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import torch
import timeit
from src.shared.components.cegis import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.shared.consts import VerifierType, TimeDomain, CertificateType
from src.plots.plot_lyap import plot_lyce


def test_lnn():
    benchmark = nonpoly0_lyap
    n_vars = 2
    system = benchmark

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [2] * len(activations)

    start = timeit.default_timer()
    opts = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.CERTIFICATE.k: CertificateType.LYAPUNOV,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.Z3,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
        CegisConfig.LLO.k: True,
    }
    c = Cegis(**opts)
    state, vars, f, iters = c.solve()
    stop = timeit.default_timer()
    print("Elapsed Time: {}".format(stop - start))

    # plotting -- only for 2-d systems
    if len(vars) == 2 and state[CegisStateKeys.found]:
        plot_lyce(
            np.array(vars), state[CegisStateKeys.V], state[CegisStateKeys.V_dot], f
        )


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()