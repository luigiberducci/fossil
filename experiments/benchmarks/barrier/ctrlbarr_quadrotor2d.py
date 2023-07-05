# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
import numpy as np

from experiments.benchmarks import models
from src import domains
from src import certificate
from src import main
from src.consts import *

# taken from Tedrake's lecture notes and the code at
# https://github.com/RussTedrake/underactuated/blob/master/underactuated/quadrotor2d.py


def test_lnn():
    batch_size = 5000
    ins = 6

    open_loop = models.Quadrotor2d

    XD = domains.Rectangle(
        lb=[-2.0, -2.0, np.pi, -2.0, -2.0, -2.0 * np.pi],
        ub=[2.0, 2.0, np.pi, 2.0, 2.0, 2.0 * np.pi],
    )

    XI = domains.Sphere([0.7, 0.7, 0.7, 0.7, 0.7, 0.7], 0.2)
    XU = domains.Sphere([-2.5] * ins, 0.2)
    XG = domains.Sphere([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0.5)

    system = models.GeneralClosedLoopModel.prepare_from_open(open_loop())

    sets = {
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
        certificate.XI: XI._generate_data(batch_size),
        certificate.XU: XU._generate_data(batch_size),
    }

    #############################
    # converges in 10 sec
    #############################

    # using trajectory control
    n_vars = 6

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [4] * len(activations)

    # control params
    n_ctrl_inputs = 2

    opts = CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.BARRIER,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        SYMMETRIC_BELT=True,
        CTRLAYER=[5, n_ctrl_inputs],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )
    main.run_benchmark(opts, repeat=1, plot=False, record=False)


if __name__ == "__main__":
    torch.manual_seed(169)
    test_lnn()
