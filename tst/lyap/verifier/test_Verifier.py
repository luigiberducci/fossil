# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
from unittest import mock

import unittest

from z3 import Reals

from src.verifier.verifier import Verifier
from src.certificate.lyapunov_certificate import LyapunovCertificate


class SimplifierTest(unittest.TestCase):
    def setUp(self) -> None:
        self.z3_vars = Reals('x y z')

    def test_whenGetSimplifiableVdot_returnSimplifiedVdot(self):
        # inputs
        x, y, z = self.z3_vars
        f = x * y + 2 * z
        domain = x*x + y*y + z*z <= 1
        return_value = 'result'
        t = 1
        lc = LyapunovCertificate(domains=[domain])

        with mock.patch.object(Verifier, '_solver_solve') as s:
            # setup
            s.return_value = return_value
            v = Verifier(3, lc.get_constraints, 0, self.z3_vars)
            v.timeout = t

            # call tested function
            res, timedout = v.solve_with_timeout(None, f)

        # assert results
        self.assertEqual(res, return_value)
        self.assertFalse(timedout)


if __name__ == '__main__':
    unittest.main()
