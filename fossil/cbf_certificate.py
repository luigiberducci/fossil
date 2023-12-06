import logging
import math
from typing import Generator

import torch
from torch.optim import Optimizer

from fossil import CegisConfig, control
from fossil.certificate import Certificate, log_loss_acc, _set_assertion
from fossil.consts import DomainNames
import fossil.learner as learner

XD = DomainNames.XD.value
XI = DomainNames.XI.value
XU = DomainNames.XU.value
XS = DomainNames.XS.value
XG = DomainNames.XG.value
XG_BORDER = DomainNames.XG_BORDER.value
XS_BORDER = DomainNames.XS_BORDER.value
XF = DomainNames.XF.value
XNF = DomainNames.XNF.value
XR = DomainNames.XR.value  # This is an override data set for ROA in StableSafe
UD = DomainNames.UD.value
HAS_BORDER = (XG, XS)
BORDERS = (XG_BORDER, XS_BORDER)
ORDER = (XD, XI, XU, XS, XG, XG_BORDER, XS_BORDER, XF, XNF)
class ControlBarrierFunction(Certificate):
    """
    Certifies Safety for continuous time controlled systems with control affine dynamics.

    Note: CBF use different conventions.
    B(Xi)>0, B(Xu)<0, Bdot(Xd) > -alpha(B(Xd)) for alpha class-k function

    Arguments:
        domains {dict}: dictionary of (string,domain) pairs
        config {CegisConfig}: configuration dictionary
    """

    def __init__(self, domains, config: CegisConfig) -> None:
        self.x_domain = domains[XD]
        self.u_domain = domains[UD]
        self.initial_domain = domains[XI]
        self.unsafe_domain = domains[XU]

        assert isinstance(config.SYSTEM, control.ControlAffineControllableDynamicalModel), "CBF only works with control-affine dynamics"
        self.fx = config.SYSTEM.fx_torch
        self.gx = config.SYSTEM.gx_torch

        # loss parameters
        self.loss_relu = torch.relu #torch.nn.Softplus()
        self.margin = 0.0
        self.epochs = 1000

    def compute_loss(
        self,
        B_i: torch.Tensor,
        B_u: torch.Tensor,
        B_d: torch.Tensor,
        Bdot_d: torch.Tensor,
        alpha: torch.Tensor | float,
    ) -> tuple[torch.Tensor, dict]:
        """Computes loss function for CBF and its accuracy w.r.t. the batch of data.

        Args:
            B_i (torch.Tensor): Barrier values for initial set
            B_u (torch.Tensor): Barrier values for unsafe set
            B_d (torch.Tensor): Barrier values for domain
            Bdot_d (torch.Tensor): Barrier derivative values for domain
            alpha (torch.Tensor): coeff. linear class-k function, f(x) = alpha * x, for alpha in R_+

        Returns:
            tuple[torch.Tensor, float]: loss and accuracy
        """
        assert Bdot_d is None or B_d.shape == Bdot_d.shape, f"B_d and Bdot_d must have the same shape, got {B_d.shape} and {Bdot_d.shape}"
        margin = self.margin

        accuracy_i = (B_i >= margin).count_nonzero().item()
        accuracy_u = (B_u < -margin).count_nonzero().item()
        accuracy_d = 0.0 #(Bdot_d + alpha * B_d >= margin).count_nonzero().item()
        percent_accuracy_init = 100 * accuracy_i / B_i.shape[0]
        percent_accuracy_unsafe = 100 * accuracy_u / B_u.shape[0]
        percent_accuracy_belt = 0.0 #100 * accuracy_d / Bdot_d.shape[0]

        relu = self.loss_relu
        init_loss = (relu(margin - B_i)).mean()     # penalize B_i < 0
        unsafe_loss = (relu(B_u + margin)).mean()   # penalize B_u > 0
        lie_loss = 0.0 #(relu(margin - Bdot_d + alpha * B_d)).mean()   # penalize dB_d + alpha * B_d < 0

        loss = init_loss + unsafe_loss + lie_loss
        accuracy = {
            "acc init": percent_accuracy_init,
            "acc unsafe": percent_accuracy_unsafe,
            "acc derivative": percent_accuracy_belt,
        }

        # debug
        print("\n".join([f"{k}:{v}" for k, v in accuracy.items()]))

        return loss, accuracy

    def learn(
        self,
        learner: learner.LearnerNN,
        optimizer: Optimizer,
        S: dict,
        Sdot: dict,
        f_torch=None,
    ) -> dict:
        """
        :param learner: learner object
        :param optimizer: torch optimiser
        :param S: dict of tensors of data
        :param Sdot: dict of tensors containing f(data)
        :return: --
        """
        assert len(S) == len(Sdot), f"expected same keys for S, Sdot. Got {S.keys()} and {Sdot.keys()}"

        condition_old = False
        i1 = S[XD].shape[0]
        i2 = S[XI].shape[0]
        # samples = torch.cat([s for s in S.values()])
        label_order = [XD, XI, XU]
        samples = torch.cat([S[label] for label in label_order])

        samples_dot = None
        #if f_torch:
        #    samples_dot = f_torch(samples)
        #else:
        #    samples_dot = torch.cat([Sdot[label] for label in label_order])

        for t in range(self.epochs):
            optimizer.zero_grad()

            #if f_torch:
            #    samples_dot = f_torch(samples)

            # net gradient
            nn, grad_nn = learner.compute_net_gradnet(samples)
            B, gradB = learner.compute_V_gradV(nn, grad_nn, samples)
            #Bdot = learner.compute_dV(gradB, Sdot)

            B_d = B[:i1]
            Bdot_d = None
            B_i = B[i1 : i1 + i2]
            B_u = B[i1 + i2 :]

            loss, accuracy = self.compute_loss(B_i, B_u, B_d, Bdot_d, alpha=1.0)

            if t % math.ceil(self.epochs / 10) == 0 or self.epochs - t < 10:
                log_loss_acc(t, loss, accuracy, learner.verbose)

            # early stopping after 2 consecutive epochs with ~100% accuracy
            condition = all(acc >= 99.9 for name, acc in accuracy.items())
            if condition and condition_old:
                break
            condition_old = condition

            loss.backward()
            optimizer.step()

        return {}

    def get_constraints(self, verifier, B, Bdot) -> Generator:
        """
        :param verifier: verifier object
        :param B: symbolic formula of the CBF
        :param Bdot: symbolic formula of the CBF derivative (not yet Lie derivative)
        :return: tuple of dictionaries of Barrier conditons
        """
        _And = verifier.solver_fncts()["And"]
        _Or = verifier.solver_fncts()["Or"]
        _Not = verifier.solver_fncts()["Not"]
        _Exists = verifier.solver_fncts()["Exists"]

        # Bdot + alpha * B >= 0
        # counterexample: Bdot + alpha * B < 0
        # lie_constr = And(B >= -0.05, B <= 0.05, Bdot > 0)
        # lie_constr = _Not(_Or(Bdot < 0, _Not(B==0)))
        #lie_constr = _And(B == 0, Bdot >= 0)

        # B >= 0 if x \in initial
        # counterexample: B < 0 and x \in initial
        initial_constr = _And(B < 0, self.initial_domain)

        # B < 0 if x \in unsafe
        # counterexample: B >= 0 and x \in unsafe
        unsafe_constr = _And(B >= 0, self.unsafe_domain)

        # add domain constraints
        #lie_constr = _And(lie_constr, self.x_domain)
        inital_constr = _And(initial_constr, self.x_domain)
        unsafe_constr = _And(unsafe_constr, self.x_domain)

        for cs in (
            {XI: inital_constr, XU: unsafe_constr},
            #{XD: lie_constr},
        ):
            yield cs

    @staticmethod
    def _assert_state(domains, data):
        domain_labels = set(domains.keys())
        data_labels = set(data.keys())
        _set_assertion(set([XD, UD, XI, XU]), domain_labels, "Symbolic Domains")
        _set_assertion(set([XD, UD, XI, XU]), data_labels, "Data Sets")

