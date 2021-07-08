# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import torch
import timeit
from src.shared.components.cegis import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, TimeDomain, CertificateType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.plots.plot_lyap import plot_lyce
from functools import partial


def test_lnn():
    batch_size = 500
    benchmark = poly_3
    n_vars = 2
    system = partial(benchmark, batch_size)

    # define domain constraints
    outer_radius = 10
    inner_radius = 0.01

    # define NN parameters
    activations = [ActivationType.SOFTPLUS, ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)

    opts = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.CERTIFICATE.k: CertificateType.LYAPUNOV,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
        CegisConfig.SP_HANDLE.k: False,
        CegisConfig.INNER_RADIUS.k: inner_radius,
        CegisConfig.OUTER_RADIUS.k: outer_radius,
        CegisConfig.LLO.k: True,
    }
    start = timeit.default_timer()
    c = Cegis(**opts)
    state, vars, f_learner, iters = c.solve()
    stop = timeit.default_timer()
    print('Elapsed Time: {}'.format(stop - start))

    # plotting -- only for 2-d systems
    if len(vars) == 2 and state[CegisStateKeys.found]:
        plot_lyce(np.array(vars), state[CegisStateKeys.V],
                      state[CegisStateKeys.V_dot], f_learner)


if __name__ == '__main__':
    torch.manual_seed(167)
    test_lnn()