# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import torch
import numpy as np
import sympy as sp
import logging

from src.shared.cegis_values import CegisStateKeys, CegisConfig, CegisComponentsState
from src.lyap.verifier.drealverifier import DRealVerifier
from src.shared.consts import LearnerType, VerifierType, ConsolidatorType, TranslatorType
from src.lyap.verifier.z3verifier import Z3Verifier
from src.shared.components.Consolidator import Consolidator
from src.shared.components.Translator import Translator
from src.lyap.utils import print_section, vprint
from src.lyap.learner.net import NN
from functools import partial

try:
    import dreal as dr
except Exception as e:
    logging.exception('Exception while importing dReal')


class Cegis:
    # todo: set params for NN and avoid useless definitions
    def __init__(self, **kw):
        self.n = kw[CegisConfig.N_VARS.k]
        # components type
        self.learner_type = kw[CegisConfig.LEARNER.k]
        self.verifier_type = kw[CegisConfig.VERIFIER.k]
        self.consolidator_type = kw[CegisConfig.CONSOLIDATOR.k]
        self.translator_type = kw[CegisConfig.TRANSLATOR.k]
        # benchmark opts
        self.inner = kw[CegisConfig.INNER_RADIUS.k]
        self.outer = kw[CegisConfig.OUTER_RADIUS.k]
        self.h = kw[CegisConfig.N_HIDDEN_NEURONS.k]
        self.activations = kw[CegisConfig.ACTIVATION.k]
        self.system = kw[CegisConfig.SYSTEM.k]
        self.sp_simplify = kw.get(CegisConfig.SP_SIMPLIFY.k, CegisConfig.SP_SIMPLIFY.v)
        self.sp_handle = kw.get(CegisConfig.SP_HANDLE.k, CegisConfig.SP_HANDLE.v)
        self.fcts = kw.get(CegisConfig.FACTORS.k, CegisConfig.FACTORS.v)
        self.eq = kw.get(CegisConfig.EQUILIBRIUM.k, CegisConfig.EQUILIBRIUM.v[0](self.n))
        self.llo = kw.get(CegisConfig.LLO.k, CegisConfig.LLO.v)
        self.rounding = kw.get(CegisConfig.ROUNDING.k, CegisConfig.ROUNDING.v)
        # other opts
        self.max_cegis_iter = kw.get(CegisConfig.CEGIS_MAX_ITERS.k, CegisConfig.CEGIS_MAX_ITERS.v)
        self.verbose = kw.get(CegisConfig.VERBOSE.k, CegisConfig.VERBOSE.v)
        # batch init
        self.learning_rate = kw.get(CegisConfig.LEARNING_RATE.k, CegisConfig.LEARNING_RATE.v)

        if self.verifier_type == VerifierType.Z3:
            verifier = Z3Verifier
        elif self.verifier_type == VerifierType.DREAL:
            verifier = DRealVerifier
        else:
            raise ValueError('No verifier of type {}'.format(self.verifier_type))

        self.x = verifier.new_vars(self.n)
        self.x_map = {str(x): x for x in self.x}

        self.f, self.f_whole_domain, self.S_d = \
            self.system(functions=verifier.solver_fncts(), inner=self.inner, outer=self.outer)
        # self.S_d = self.S_d.requires_grad_(True)

        # self.verifier = verifier(self.n, self.domain, self.initial_s, self.unsafe, vars_bounds, self.x)
        self.domain = self.f_whole_domain(verifier.solver_fncts(), self.x)
        self.verifier = verifier(self.n, self.eq, self.domain, self.x, **kw)

        self.xdot = self.f(self.verifier.solver_fncts(), self.x)
        self.x = np.matrix(self.x).T
        self.xdot = np.matrix(self.xdot).T

        if self.learner_type == LearnerType.NN:
            self.learner = NN(self.n, *self.h, bias=False, activate=self.activations,
                              equilibria=self.eq, llo=self.llo, **kw)
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)
        else:
            raise ValueError('No learner of type {}'.format(self.learner_type))

        self.f_verifier = partial(self.f, self.verifier.solver_fncts())
        self.f_learner = partial(self.f, self.learner.learner_fncts())

        if self.sp_handle:
            self.x = [sp.Symbol('x%d' % i, real=True) for i in range(self.n)]
            self.xdot = self.f({'sin': sp.sin, 'cos': sp.cos, 'exp': sp.exp}, self.x)
            self.x_map = {**self.x_map, **self.verifier.solver_fncts()}
            self.x, self.xdot = np.matrix(self.x).T, np.matrix(self.xdot).T
        else:
            self.x_sympy, self.xdot_s = None, None

        if self.consolidator_type == ConsolidatorType.DEFAULT:
            self.consolidator = Consolidator(self.f_learner)
        else:
            TypeError('Not Implemented Consolidator')
        if self.translator_type == TranslatorType.DEFAULT:
            self.translator = Translator(self.learner, self.x, self.xdot, self.eq, self.rounding, **kw)
        else:
            TypeError('Not Implemented Translator')

        self._result = None

    # the cegis loop
    # todo: fix return, fix map(f, S)
    def solve(self):

        # Sdot = self.f_learner(self.S_d.T)
        # needed to make hybrid work
        Sdot = list(map(torch.tensor, map(self.f_learner, self.S_d)))
        S, Sdot = self.S_d, torch.stack(Sdot)

        if self.learner_type == LearnerType.NN:
            self.optimizer = torch.optim.AdamW(self.learner.parameters(), lr=self.learning_rate)

        stats = {}
        # the CEGIS loop
        iters = 0
        stop = False

        components = [
            {
                CegisComponentsState.name: 'learner',
                CegisComponentsState.instance: self.learner,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: 'translator',
                CegisComponentsState.instance: self.translator,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: 'verifier',
                CegisComponentsState.instance: self.verifier,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: 'consolidator',
                CegisComponentsState.instance: self.consolidator,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw
            }
        ]

        state = {
            CegisStateKeys.net: self.learner,
            CegisStateKeys.optimizer: self.optimizer,
            CegisStateKeys.S: S,
            CegisStateKeys.S_dot: Sdot,
            CegisStateKeys.factors: self.fcts,
            CegisStateKeys.sp_handle: self.sp_handle,
            CegisStateKeys.V: None,
            CegisStateKeys.V_dot: None,
            CegisStateKeys.x_v_map: self.x_map,
            CegisStateKeys.verifier_fun: self.f_verifier,
            CegisStateKeys.found: False,
            CegisStateKeys.verification_timed_out: False,
            CegisStateKeys.cex: None,
            CegisStateKeys.trajectory: None
        }

        # reset timers
        self.learner.get_timer().reset()
        self.translator.get_timer().reset()
        self.verifier.get_timer().reset()
        self.consolidator.get_timer().reset()

        while not stop:
            for component_idx in range(len(components)):
                component = components[component_idx]
                next_component = components[(component_idx + 1) % len(components)]

                if self.verbose:
                    print_section(component[CegisComponentsState.name], iters)
                outputs = component[CegisComponentsState.instance].get(**state)

                state = {**state, **outputs}

                state = {**state, **(
                    component[CegisComponentsState.to_next_component](
                        outputs, next_component[CegisComponentsState.instance], **state
                    ))}

                if state[CegisStateKeys.found] and component_idx == len(components)-1:
                    print('Found a Lyapunov function')
                    stop = True
                if state[CegisStateKeys.verification_timed_out]:
                    print('Verification Timed Out')
                    stop = True

            if self.max_cegis_iter == iters and not state[CegisStateKeys.found]:
                print('Out of Cegis loops')
                stop = True

            iters += 1
            if not (state[CegisStateKeys.found] or state[CegisStateKeys.verification_timed_out]):
                self.learner.find_closest_unsat(state[CegisStateKeys.S], state[CegisStateKeys.S_dot], self.fcts)
                # S, Sdot = self.add_ces_to_data(S, Sdot, ces)
                state[CegisStateKeys.S], state[CegisStateKeys.S_dot] = \
                    self.add_ces_to_data(state[CegisStateKeys.S], state[CegisStateKeys.S_dot],
                                         torch.cat((state[CegisStateKeys.cex], state[CegisStateKeys.trajectory])))

        state[CegisStateKeys.components_times] = [
            self.learner.get_timer().sum, self.translator.get_timer().sum,
            self.verifier.get_timer().sum, self.consolidator.get_timer().sum
        ]
        vprint(['Learner times: {}'.format(self.learner.get_timer())], self.verbose)
        vprint(['Translator times: {}'.format(self.translator.get_timer())], self.verbose)
        vprint(['Verifier times: {}'.format(self.verifier.get_timer())], self.verbose)
        vprint(['Consolidator times: {}'.format(self.consolidator.get_timer())], self.verbose)

        self._result = state, self.x, self.f_learner, iters
        return self._result

    @property
    def result(self):
        return self._result

    def add_ces_to_data(self, S, Sdot, ces):
        """
        :param S: torch tensor
        :param Sdot: torch tensor
        :param ces: list of ctx
        :return:
                S: torch tensor, added new ctx
                Sdot torch tensor, added  f(new_ctx)
        """
        S = torch.cat([S, ces], dim=0).detach()
        # Sdot = torch.stack(self.f_learner(S.T)).T
        # needed to make hybrid work
        Sdot = torch.stack(list(map(torch.tensor, map(self.f_learner, S))))
        # torch.cat([Sdot, torch.stack(self.f_learner(ces.T)).T], dim=0)
        return S, Sdot
