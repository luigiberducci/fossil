import traceback
from functools import partial

import torch
import numpy as np
import pandas as pd
import timeit

from experiments.benchmarks.benchmarks_bc import barr_1
from src.barrier.cegis_barrier import Cegis
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.shared.consts import VerifierType, LearnerType, TrajectoriserType, RegulariserType


def test_robustness(h):
    batch_size = 500
    system = partial(barr_1, batch_size)
    activations = [ActivationType.LINEAR, ActivationType.LIN_TO_CUBIC, ActivationType.LINEAR]
    hidden_neurons = [h] * len(activations)

    start = timeit.default_timer()
    opts = {
        CegisConfig.N_VARS.k: 2,
        CegisConfig.LEARNER.k: LearnerType.NN,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.TRAJECTORISER.k: TrajectoriserType.DEFAULT,
        CegisConfig.REGULARISER.k: RegulariserType.DEFAULT,
        CegisConfig.ACTIVATION.k: activations,
        CegisConfig.SYSTEM.k: system,
        CegisConfig.N_HIDDEN_NEURONS.k: hidden_neurons,
        CegisConfig.SP_SIMPLIFY.k: True,
    }
    c = Cegis(**opts)
    state, vars, f_learner, iters = c.solve()
    end = timeit.default_timer()

    return end-start, state[CegisStateKeys.found], state['components_times'], iters


if __name__ == '__main__':
    number_of_runs = 100

    # we advise you to comment out all the print options
    for hidden in [2, 10, 50, 100]:
        res = pd.DataFrame(columns=['found', 'iters', 'elapsed_time',
                                    'lrn_time', 'reg_time', 'ver_time', 'trj_time'])

        for idx in range(number_of_runs):
            el_time, found, comp_times, iters = test_robustness(h=hidden)
            res = res.append({'found': found, 'iters': iters,
                              'elapsed_time': el_time,
                              'lrn_time': comp_times[0], 'reg_time': comp_times[1],
                              'ver_time': comp_times[2], 'trj_time': comp_times[3]},
                             ignore_index=True)

        name_save = 'robustness_barrier' + '_hdn_' + str(hidden) + '.csv'
        res.to_csv(name_save)

