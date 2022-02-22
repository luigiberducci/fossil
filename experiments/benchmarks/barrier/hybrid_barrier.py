# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
# pylint: disable=not-callable
from experiments.benchmarks.benchmarks_bc import twod_hybrid
from src.shared.components.cegis import Cegis
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, TimeDomain, CertificateType
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig, CegisStateKeys

from functools import partial
import timeit
import torch


def main():

    system = twod_hybrid
    activations = [ActivationType.LIN_SQUARE]
    hidden_neurons = [3] * len(activations)
    start = timeit.default_timer()
    opts = {
        CegisConfig.N_VARS.k: 2,
        CegisConfig.CERTIFICATE.k: CertificateType.BARRIER,
        CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
        CegisConfig.VERIFIER.k: VerifierType.Z3,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
        CegisConfig.SYMMETRIC_BELT.k: False,
    }
    c = Cegis(**opts)
    state, vars, f_learner, iters = c.solve()
    end = timeit.default_timer()

    print('Elapsed Time: {}'.format(end - start))
    print("Found? {}".format(state[CegisStateKeys.found]))


if __name__ == '__main__':
    torch.manual_seed(167)
    main()