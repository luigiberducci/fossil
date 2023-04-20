# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable

import torch
import timeit

from experiments.benchmarks.benchmarks_bc import barr_2
from src.shared.components.cegis import Cegis

from src.shared.consts import *

import src.plots.plot_fcns as plotting
import numpy as np


def main():
    ###############################
    # takes 4.6 secs, at iter 3
    ###############################

    system = barr_2
    activations = [ActivationType.TANH]
    hidden_neurons = [8]
    opts = CegisConfig(
        N_VARS=2,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
    )

    start = timeit.default_timer()
    c = Cegis(opts)
    state, vars, f, iters = c.solve()
    end = timeit.default_timer()

    print("Elapsed Time: {}".format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))

    plotting.benchmark(f, c.learner, {}, levels=[0])

    # plotting -- only for 2-d systems
    # if state[CegisStateKeys.found]:
    #     plot_exponential_bench(np.array(vars), state[CegisStateKeys.V])


if __name__ == "__main__":
    torch.manual_seed(167)
    main()
