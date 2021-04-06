# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import unittest
from unittest import mock
from functools import partial
from src.lyap.learner.net import NN
from src.shared.components.Consolidator import Consolidator
from src.shared.activations import ActivationType
from experiments.benchmarks.benchmarks_lyap import poly_2
from src.shared.cegis_values import CegisStateKeys
import torch


class ConsolidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.n_vars = 2
        system = partial(poly_2, batch_size=500)
        self.f, _, self.S_d = system(functions={'And': 0})
        self.f_learner = partial(self.f, {'And': 0})
        self.hidden = [3]
        self.activate = [ActivationType.SQUARE]

    # given a point, the consolidator returns a list of points - trajectory -
    # that lead towards the max of Vdot
    def test_fromCex_returnTrajectoryTowardsHighestValueOfVdot(self):
        # give a value to a hypothetical cex
        point = torch.tensor([1., 2.])

        # def neural learner
        with mock.patch.object(NN, 'learn') as learner:
            # setup learner
            learner.input_size = 2
            learner.acts = [ActivationType.SQUARE]
            learner.layers = [
                torch.nn.Linear(2, 3, bias=False),
                torch.nn.Linear(3, 1, bias=False)
            ]
            learner.layers[0].weight = torch.nn.Parameter(torch.tensor(
                [[1.0, 2.0],
                 [2.0, 1.0],
                 [5.0, 4.0]
            ]))
            learner.layers[1].weight = torch.nn.Parameter(
                torch.tensor([-1.0, 1.0, -2.0]).reshape(1, 3)
            )

            # create a 'real' consolidator
            traj = Consolidator(self.f_learner)
            state = {
                CegisStateKeys.net: learner,
                CegisStateKeys.cex: [point],
                CegisStateKeys.trajectory: None,
            }
            output = traj.get(**state)
            state = {**state, **output}
            trajectory = state[CegisStateKeys.trajectory]
            # evaluate the points in Vdot(trajectory)
            v_dots = []
            for idx in range(len(trajectory)):
                v_dots.append(traj.forward_Vdot(learner, trajectory[idx].detach()).item())

            # check that Vdot(trajectory) is an increasing sequence
            self.assertTrue(all(v_dots[i] <= v_dots[i+1] for i in range(len(v_dots)-1)))


if __name__ == '__main__':
    unittest.main()