# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from itertools import chain

import numpy as np
import torch

from experiments.benchmarks.models import ClosedLoopModel, GeneralClosedLoopModel
import src.certificate as certificate
import src.learner as learner
from src.shared.components.consolidator import Consolidator
import src.shared.control as control
from src.shared.consts import *
from src.shared.utils import print_section, vprint
import src.translator as translator
import src.verifier as verifier


class Cegis:
    # todo: set params for NN and avoid useless definitions
    def __init__(self, config: CegisConfig):
        self.config = config

        self.learner_type = learner.get_learner(
            self.config.TIME_DOMAIN, self.config.CTRLAYER
        )
        self.translator_type = translator.get_translator_type(
            self.config.TIME_DOMAIN, self.config.VERIFIER
        )

        # Verifier init
        verifier_type = verifier.get_verifier_type(self.config.VERIFIER)

        self.x = verifier_type.new_vars(self.config.N_VARS)
        self.x_map = {str(x): x for x in self.x}

        # System init
        self.system = self.config.SYSTEM
        # if controller, initialise system with the controller
        if self.config.CTRLAYER:
            # todo
            # ctrler = GeneralController(ctrl_layers)  --> pass to self.system
            ctrl_activ = self.config.CTRLACTIVATION
            self.ctrler = control.GeneralController(
                inputs=self.config.N_VARS,
                output=self.config.CTRLAYER[-1],
                layers=self.config.CTRLAYER[:-1],
                activations=ctrl_activ,
            )
            self.f, self.f_domains, self.S, vars_bounds = self.system(self.ctrler)
        else:
            self.f, self.f_domains, self.S, vars_bounds = self.system()

        # Overwrite domains if provided
        # This is a precursor to providing the sets separately to CEGIS, rather than in bulk with the model
        # self.f_domains = kw.get(CegisConfig.XD.k, self.f_domains)
        # self.S = kw.get(CegisConfig.SD.k, self.S)

        self.domains = {
            label: domain(self.x) for label, domain in self.f_domains.items()
        }
        certificate_type = certificate.get_certificate(self.config.CERTIFICATE)
        if self.config.CERTIFICATE == certificate.CertificateType.STABLESAFE:
            raise ValueError("StableSafe not compatible with default CEGIS")
        self.certificate = certificate_type(self.domains, self.config)

        self.verifier = verifier.get_verifier(
            verifier_type,
            self.config.N_VARS,
            self.certificate.get_constraints,
            vars_bounds,
            self.x,
            self.config.VERBOSE,
        )

        self.xdot = self.f(self.x)

        # Learner init
        self.learner = self.learner_type(
            self.config.N_VARS,
            self.certificate.learn,
            *self.config.N_HIDDEN_NEURONS,
            activation=self.config.ACTIVATION,
            bias=self.certificate.bias,
            config=self.config,
        )

        self.optimizer = torch.optim.AdamW(
            chain(self.learner.parameters(), self.f.parameters),
            lr=self.config.LEARNING_RATE,
        )

        if self.config.CONSOLIDATOR == ConsolidatorType.DEFAULT:
            self.consolidator = Consolidator(self.f)
        else:
            TypeError("Not Implemented Consolidator")

        # Translator init
        self.translator = translator.get_translator(
            self.translator_type,
            # self.learner,
            self.x,
            self.xdot,
            self.config.EQUILIBRIUM,
            self.config.ROUNDING,
            verbose=self.config.VERBOSE,
        )
        self._result = None

    def solve(self):

        Sdot = {lab: self.f(S) for lab, S in self.S.items()}
        S = self.S

        # the CEGIS loop
        iters = 0
        stop = False

        components = [
            {
                CegisComponentsState.name: "learner",
                CegisComponentsState.instance: self.learner,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: "translator",
                CegisComponentsState.instance: self.translator,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: "verifier",
                CegisComponentsState.instance: self.verifier,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: "consolidator",
                CegisComponentsState.instance: self.consolidator,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
        ]

        state = {
            CegisStateKeys.net: self.learner,
            CegisStateKeys.optimizer: self.optimizer,
            CegisStateKeys.S: S,
            CegisStateKeys.S_dot: Sdot,
            CegisStateKeys.factors: self.config.FACTORS,
            CegisStateKeys.V: None,
            CegisStateKeys.V_dot: None,
            CegisStateKeys.x_v_map: self.x_map,
            CegisStateKeys.xdot: self.xdot,
            CegisStateKeys.xdot_func: self.f.f_torch,
            CegisStateKeys.found: False,
            CegisStateKeys.verification_timed_out: False,
            CegisStateKeys.cex: None,
            CegisStateKeys.trajectory: None,
            CegisStateKeys.ENet: self.config.ENET,
        }

        # reset timers
        self.learner.get_timer().reset()
        self.translator.get_timer().reset()
        self.verifier.get_timer().reset()
        self.consolidator.get_timer().reset()

        while not stop:
            for component_idx in range(len(components)):
                if component_idx == 1:
                    # Update controller before translation
                    state.update({CegisStateKeys.xdot: self.f(self.x)})
                component = components[component_idx]
                next_component = components[(component_idx + 1) % len(components)]

                if self.config.VERBOSE:
                    print_section(component[CegisComponentsState.name], iters)
                outputs = component[CegisComponentsState.instance].get(**state)

                state = {**state, **outputs}

                state = {
                    **state,
                    **(
                        component[CegisComponentsState.to_next_component](
                            outputs,
                            next_component[CegisComponentsState.instance],
                            **state,
                        )
                    ),
                }

                if state[CegisStateKeys.found] and component_idx == len(components) - 1:
                    if self.config.CERTIFICATE == CertificateType.RSWA:
                        stop = self.certificate.stay_in_goal_check(
                            self.verifier,
                            state[CegisStateKeys.V],
                            state[CegisStateKeys.V_dot],
                        )
                        if stop:
                            print(
                                f"Found a valid {self.config.CERTIFICATE.name} certificate"
                            )
                    else:
                        print(
                            f"Found a valid {self.config.CERTIFICATE.name} certificate"
                        )
                        stop = True

                if state[CegisStateKeys.verification_timed_out]:
                    print("Verification Timed Out")
                    stop = True

            if self.config.CEGIS_MAX_ITERS == iters and not state[CegisStateKeys.found]:
                print("Out of Cegis loops")
                stop = True

            iters += 1
            if not (
                state[CegisStateKeys.found]
                or state[CegisStateKeys.verification_timed_out]
            ):
                if state[CegisStateKeys.trajectory] != []:
                    lie_label = [key for key in S.keys() if "lie" in key][0]
                    state[CegisStateKeys.cex][lie_label] = torch.cat(
                        [
                            state[CegisStateKeys.cex][lie_label],
                            state[CegisStateKeys.trajectory],
                        ]
                    )
                (
                    state[CegisStateKeys.S],
                    state[CegisStateKeys.S_dot],
                ) = self.add_ces_to_data(
                    state[CegisStateKeys.S],
                    state[CegisStateKeys.S_dot],
                    state[CegisStateKeys.cex],
                )
                if isinstance(self.f, ClosedLoopModel) or isinstance(
                    self.f, GeneralClosedLoopModel
                ):
                    pass
                    # self.f.plot()
                    # It might be better to have a CONTROLLED param to cegis, but there's
                    # already a lot of those so tried to avoid that.
                    # optim = torch.optim.AdamW(self.f.controller.parameters())
                    # self.f.controller.learn(
                    #     state[CegisStateKeys.S][self.certificate.XD],
                    #     self.f.open_loop,
                    #     optim,
                    # )
                    # state.update({CegisStateKeys.xdot: self.f(self.x)})

        state[CegisStateKeys.components_times] = [
            self.learner.get_timer().sum,
            self.translator.get_timer().sum,
            self.verifier.get_timer().sum,
            self.consolidator.get_timer().sum,
        ]
        vprint(
            ["Learner times: {}".format(self.learner.get_timer())], self.config.VERBOSE
        )
        vprint(
            ["Translator times: {}".format(self.translator.get_timer())],
            self.config.VERBOSE,
        )
        vprint(
            ["Verifier times: {}".format(self.verifier.get_timer())],
            self.config.VERBOSE,
        )
        vprint(
            ["Consolidator times: {}".format(self.consolidator.get_timer())],
            self.config.VERBOSE,
        )

        self._result = state, np.array(self.x).reshape(-1, 1), self.f, iters
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
        for lab, cex in ces.items():
            if cex != []:
                S[lab] = torch.cat([S[lab], cex], dim=0).detach()
                Sdot[lab] = self.f(S[lab])
        return S, Sdot

    def _assert_state(self):
        assert self.verifier_type in [
            VerifierType.Z3,
            VerifierType.DREAL,
            VerifierType.MARABOU,
        ]
        assert self.learner_type in [LearnerType.CONTINUOUS, LearnerType.DISCRETE]
        assert self.batch_size > 0
        assert self.learning_rate > 0
        assert self.max_cegis_time > 0


class DoubleCegis(Cegis):
    """StableSafe Cegis in parallel.

    A stable while stay criterion relies on an open set D, compact sets XI, XG and a closed set XU.
    http://arxiv.org/abs/2009.04432, https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9483376.

    Necessarily there exists A \subset G. Goal is to synth two smooth functions V, B such that:

    (1) V is positive definite wrt A (V(x) = 0 iff x \in A)
    (2) \forall x in D \ A: dV/dt < 0
    (3) \forall x \in XI, B(x) >= 0; \forall x in XU: B(x) <0
    (4) \forall x \in D: dB/dt >= 0"""

    def __init__(self, config: CegisConfig):
        self.config = config

        self.learner_type = learner.get_learner(
            self.config.TIME_DOMAIN, self.config.CTRLAYER
        )
        self.translator_type = translator.get_translator_type(
            self.config.TIME_DOMAIN, self.config.VERIFIER
        )

        # Verifier init
        verifier_type = verifier.get_verifier_type(self.config.VERIFIER)

        self.x = verifier_type.new_vars(self.config.N_VARS)
        self.x_map = {str(x): x for x in self.x}

        # System init
        self.system = self.config.SYSTEM
        # if controller, initialise system with the controller
        if self.config.CTRLAYER:
            # todo
            # ctrler = GeneralController(ctrl_layers)  --> pass to self.system
            ctrl_activ = self.config.CTRLACTIVATION
            self.ctrler = control.GeneralController(
                inputs=self.config.N_VARS,
                output=self.config.CTRLAYER[-1],
                layers=self.config.CTRLAYER[:-1],
                activations=ctrl_activ,
            )
            self.f, self.f_domains, self.S, vars_bounds = self.system(self.ctrler)
        else:
            self.f, self.f_domains, self.S, vars_bounds = self.system()

        # Overwrite domains if provided
        # This is a precursor to providing the sets separately to CEGIS, rather than in bulk with the model
        # self.f_domains = kw.get(CegisConfig.XD.k, self.f_domains)
        # self.S = kw.get(CegisConfig.SD.k, self.S)

        self.domains = {
            label: domain(self.x) for label, domain in self.f_domains.items()
        }
        certificate_type = certificate.get_certificate(self.config.CERTIFICATE)
        if self.config.CERTIFICATE != CertificateType.STABLESAFE:
            raise ValueError("DoubleCegis only supports StableSafe certificates")
        self.certificate = certificate_type(self.domains, self.config)

        self.verifier = verifier.get_verifier(
            verifier_type,
            self.config.N_VARS,
            self.certificate.get_constraints,
            vars_bounds,
            self.x,
            self.config.VERBOSE,
        )

        self.xdot = self.f(self.x)

        # Learner init
        self.lyap_learner = self.learner_type(
            self.config.N_VARS,
            self.certificate.learn,
            *self.config.N_HIDDEN_NEURONS,
            bias=False,
            activation=self.config.ACTIVATION,
            config=self.config,
        )

        self.barr_learner = self.learner_type(
            self.config.N_VARS,
            self.certificate.learn,
            *self.config.N_HIDDEN_NEURONS_ALT,
            bias=True,
            activation=self.config.ACTIVATION_ALT,
            config=self.config,
        )

        if self.config.CONSOLIDATOR == ConsolidatorType.DEFAULT:
            self.consolidator = Consolidator(self.f)
        else:
            TypeError("Not Implemented Consolidator")

        # Translator init
        self._result = None

        self.optimizer = torch.optim.AdamW(
            chain(
                self.lyap_learner.parameters(),
                self.barr_learner.parameters(),
                self.f.parameters,
            ),
            lr=self.config.LEARNING_RATE,
        )

        # Translator init
        self.translator_type = translator.TranslatorCTDouble
        self.translator = translator.get_translator(
            self.translator_type,
            self.x,
            self.xdot,
            self.config.EQUILIBRIUM,
            self.config.ROUNDING,
            verbose=self.config.VERBOSE,
        )

    def solve(self):

        Sdot = {lab: self.f(S) for lab, S in self.S.items()}
        S = self.S

        # the CEGIS loop
        iters = 0
        stop = False

        components = [
            {
                CegisComponentsState.name: "learner",
                CegisComponentsState.instance: self.lyap_learner,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: "translator",
                CegisComponentsState.instance: self.translator,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            {
                CegisComponentsState.name: "verifier",
                CegisComponentsState.instance: self.verifier,
                CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            },
            # {
            #     CegisComponentsState.name: "consolidator",
            #     CegisComponentsState.instance: self.consolidator,
            #     CegisComponentsState.to_next_component: lambda _outputs, next_component, **kw: kw,
            # },
        ]

        state = {
            CegisStateKeys.net: (self.lyap_learner, self.barr_learner),
            CegisStateKeys.optimizer: self.optimizer,
            CegisStateKeys.S: S,
            CegisStateKeys.S_dot: Sdot,
            CegisStateKeys.factors: self.config.FACTORS,
            CegisStateKeys.V: None,
            CegisStateKeys.V_dot: None,
            CegisStateKeys.x_v_map: self.x_map,
            CegisStateKeys.xdot: self.xdot,
            CegisStateKeys.xdot_func: self.f.f_torch,
            CegisStateKeys.found: False,
            CegisStateKeys.verification_timed_out: False,
            CegisStateKeys.cex: None,
            CegisStateKeys.trajectory: None,
            CegisStateKeys.ENet: self.config.ENET,
        }

        # reset timers
        self.lyap_learner.get_timer().reset()
        self.translator.get_timer().reset()
        self.verifier.get_timer().reset()
        self.consolidator.get_timer().reset()

        while not stop:
            for component_idx in range(len(components)):
                if component_idx == 1:
                    # Update controller before translation
                    state.update({CegisStateKeys.xdot: self.f(self.x)})
                component = components[component_idx]
                next_component = components[(component_idx + 1) % len(components)]

                if self.config.VERBOSE:
                    print_section(component[CegisComponentsState.name], iters)
                outputs = component[CegisComponentsState.instance].get(**state)

                state = {**state, **outputs}

                state = {
                    **state,
                    **(
                        component[CegisComponentsState.to_next_component](
                            outputs,
                            next_component[CegisComponentsState.instance],
                            **state,
                        )
                    ),
                }

                if state[CegisStateKeys.found] and component_idx == len(components) - 1:
                    if self.config.CERTIFICATE == CertificateType.RSWA:
                        stop = self.certificate.stay_in_goal_check(
                            self.verifier,
                            state[CegisStateKeys.V],
                            state[CegisStateKeys.V_dot],
                        )
                        if stop:
                            print(
                                f"Found a valid {self.config.CERTIFICATE.name} certificate"
                            )
                    else:
                        print(
                            f"Found a valid {self.config.CERTIFICATE.name} certificate"
                        )
                        stop = True

                if state[CegisStateKeys.verification_timed_out]:
                    print("Verification Timed Out")
                    stop = True

            if self.config.CEGIS_MAX_ITERS == iters and not state[CegisStateKeys.found]:
                print("Out of Cegis loops")
                stop = True

            iters += 1
            if not (
                state[CegisStateKeys.found]
                or state[CegisStateKeys.verification_timed_out]
            ):
                if state[CegisStateKeys.trajectory] != []:
                    pass
                    # lie_label = [key for key in S.keys() if "lie" in key][0]
                    # state[CegisStateKeys.cex][lie_label] = torch.cat(
                    #     [
                    #         state[CegisStateKeys.cex][lie_label],
                    #         state[CegisStateKeys.trajectory],
                    #     ]
                    # )
                (
                    state[CegisStateKeys.S],
                    state[CegisStateKeys.S_dot],
                ) = self.add_ces_to_data(
                    state[CegisStateKeys.S],
                    state[CegisStateKeys.S_dot],
                    state[CegisStateKeys.cex],
                )
                if isinstance(self.f, ClosedLoopModel) or isinstance(
                    self.f, GeneralClosedLoopModel
                ):
                    pass
                    # self.f.plot()
                    # It might be better to have a CONTROLLED param to cegis, but there's
                    # already a lot of those so tried to avoid that.
                    # optim = torch.optim.AdamW(self.f.controller.parameters())
                    # self.f.controller.learn(
                    #     state[CegisStateKeys.S][self.certificate.XD],
                    #     self.f.open_loop,
                    #     optim,
                    # )
                    # state.update({CegisStateKeys.xdot: self.f(self.x)})

        state[CegisStateKeys.components_times] = [
            self.lyap_learner.get_timer().sum,
            self.translator.get_timer().sum,
            self.verifier.get_timer().sum,
            self.consolidator.get_timer().sum,
        ]
        vprint(
            ["Learner times: {}".format(self.lyap_learner.get_timer())],
            self.config.VERBOSE,
        )
        vprint(
            ["Translator times: {}".format(self.translator.get_timer())],
            self.config.VERBOSE,
        )
        vprint(
            ["Verifier times: {}".format(self.verifier.get_timer())],
            self.config.VERBOSE,
        )
        vprint(
            ["Consolidator times: {}".format(self.consolidator.get_timer())],
            self.config.VERBOSE,
        )

        self._result = state, np.array(self.x).reshape(-1, 1), self.f, iters
        return self._result
