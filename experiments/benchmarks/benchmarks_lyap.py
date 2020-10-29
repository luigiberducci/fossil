import sympy as sp
import re
import z3 as z3
from experiments.benchmarks.domain_fcns import *
import matplotlib.pyplot as plt


###############################
### NON POLY BENCHMARKS
###############################

# this series comes from
# 2014, Finding Non-Polynomial Positive Invariants and Lyapunov Functions forPolynomial Systems through Darboux Polynomials.

# also from CDC 2011, Parrillo, poly system w non-poly lyap
def nonpoly0(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return [
            -x + x * y,
            -y
        ]

    def XD(_, v):
        x, y = v
        return _And(x > 0, y > 0,
                    inner ** 2 <= x ** 2 + y ** 2, x ** 2 + y ** 2 <= outer**2)

    def SD():
        return slice_init_data((0, 0), outer**2, batch_size)

    return f, XD, SD()


def nonpoly1(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return  [
                -x + 2*x**2 * y,
                -y
                ]

    def XD(_, v):
        x, y = v
        return _And(x > 0, y > 0, x ** 2 + y ** 2 <= outer**2)

    def SD():
        return slice_init_data((0, 0), outer**2, batch_size)

    return f, XD, SD()


def nonpoly2(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y, z = v
        return  [
                -x,
                -2*y + 0.1*x*y**2 + z,
                -z -1.5*y
                ]

    def XD(_, v):
        x, y, z = v
        return _And(x > 0, y > 0, z > 0, x ** 2 + y ** 2 + z**2 <= outer**2)

    def SD():
        return slice_3d_init_data((0, 0, 0), outer**2, batch_size)

    return f, XD, SD()


def nonpoly3(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y, z = v
        return  [
                -3*x - 0.1*x*y**3,
                -y + z,
                -z
                ]

    def XD(_, v):
        x, y, z = v
        return _And(x > 0, y > 0, z > 0, x ** 2 + y ** 2 + z**2 <= outer**2)

    def SD():
        return slice_3d_init_data((0, 0, 0), outer**2, batch_size)

    return f, XD, SD()


######################
# TACAS benchmarks
######################


def benchmark_0(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    # test function, not to be included
    def f(_, v):
        x, y = v
        return [-x, -y]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def benchmark_1(batch_size, functions, inner=0.0, outer=10.0):
    # SOSDEMO2
    # from http://sysos.eng.ox.ac.uk/sostools/sostools.pdf
    _And = functions['And']

    # test function, not to be included
    def f(_, v):
        x, y, z = v
        return [
            -x**3 - x*z**2,
            -y - x**2 * y,
            -z + 3*x**2*z - (3*z)
        ]

    def XD(_, v):
        x, y, z = v
        return _And(x ** 2 + y ** 2 + z ** 2 > inner, x ** 2 + y ** 2 + z ** 2 <= outer ** 2)

    def SD():
        return sphere_init_data((0, 0, 0), outer ** 2, batch_size)

    return f, XD, SD()


# this series comes from
# https://www.cs.colorado.edu/~srirams/papers/nolcos13.pdf
# srirams paper from 2013 (old-ish) but plenty of lyap fcns

def benchmark_3(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        # if v.shape[0] == 1:
        #     return [- v[0, 0] ** 3 + v[0, 1], - v[0, 0] - v[0, 1]]
        # else:
        #     return [- v[:, 0] ** 3 + v[:, 1], - v[:, 0] - v[:, 1]]
        x,y = v
        return [- x**3 + y, - x - y]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def benchmark_4(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return [-x**3 - y**2, x*y - y**3]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def benchmark_5(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y = v
        return [
            -x - 1.5 * x**2 * y**3,
            -y**3 + 0.5 * x**3 * y**2
        ]

    def XD(_, v):
        x, y = v
        return _And(x ** 2 + y ** 2 > inner, x ** 2 + y ** 2 <= outer ** 2)

    def SD():
        return circle_init_data((0, 0), outer ** 2, batch_size)

    return f, XD, SD()


def benchmark_6(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x, y, w, z = v
        return [-x + y**3 - 3*w*z, -x - y**3, x*z - w, x*w - z**3]

    def XD(_, v):
        x, y, w, z = v
        return _And(x ** 2 + y ** 2 + w ** 2 + z ** 2 > inner**2,
                    x ** 2 + y ** 2 + w ** 2 + z ** 2 <= outer ** 2)

    def SD():
        return n_dim_sphere_init_data((0, 0, 0, 0), outer, batch_size)

    return f, XD, SD()


def benchmark_7(batch_size, functions, inner=0.0, outer=10.0):
    _And = functions['And']

    def f(_, v):
        x0, x1, x2, x3, x4, x5 = v
        return [
            - x0 ** 3 + 4 * x1 ** 3 - 6 * x2 * x3,
            -x0 - x1 + x4 ** 3,
            x0 * x3 - x2 + x3 * x5,
            x0 * x2 + x2 * x5 - x3 ** 3,
            - 2 * x1 ** 3 - x4 + x5,
            -3 * x2 * x3 - x4 ** 3 - x5
        ]

    def XD(_, v):
        x0, x1, x2, x3, x4, x5 = v
        return _And(x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 > inner ** 2,
                    x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 <= outer ** 2)

    def SD():
        return n_dim_sphere_init_data((0, 0, 0, 0, 0, 0), outer, batch_size)

    return f, XD, SD()


def twod_hybrid(batch_size, functions, inner, outer):
    # example of 2-d hybrid sys
    _And = functions['And']

    def f(functions, v):
        _If = functions['If']
        x0, x1 = v
        _then = - x1 - 0.5*x0**3
        _else = - x1 - x0**2 - 0.25*x1**3
        _cond = x1 >= 0
        return [-x0, _If(_cond, _then, _else)]

    def XD(_, v):
        x0, x1 = v
        return _And(inner**2 < x0**2 + x1**2,
                               x0**2 + x1**2 <= outer**2)

    def SD():
        return circle_init_data((0., 0.), outer**2, batch_size)

    return f, XD, SD()


##########################
# LINEAR HIGH DIM
##########################


# 10-d version of pj_original
def four_poly(batch_size, functions, inner=0., outer=10.):
    _And = functions['And']
    _Or = functions['Or']

    def f(_, v):
        x0, x1, x2, x3 = v
        # x^4 + 3980 x^3 + 4180 x^2 + 2400 x + 576
        # is stable with complex roots
        return [x1, x2,
                x3,
                - 3980*x3 - 4180*x2 - 2400*x1 - 576*x0
                ]

    def XD(_, v):
        x0, x1, x2, x3 = v
        return _And(inner**2 < x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2,
                    x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 <= outer**2)

    def SD():
        return n_dim_sphere_init_data([0, 0, 0, 0], outer, batch_size)

    return f, XD, SD()


# 10-d version of pj_original
def six_poly(batch_size, functions, inner=0., outer=10.):
    _And = functions['And']
    _Or = functions['Or']

    def f(_, v):
        x0, x1, x2, x3, x4, x5 = v
        # x^6 + 800 x^5 + 2273 x^4 + 3980 x^3 + 4180 x^2 + 2400 x + 576
        # is stable with complex roots
        return [x1, x2,
                x3, x4,
                x5,
                - 800*x5 - 2273*x4 - 3980*x3 - 4180*x2 - 2400*x1 - 576*x0
                ]

    def XD(_, v):
        x0, x1, x2, x3, x4, x5 = v
        return _And(inner <= x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2,
                    x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 <= outer **2)

    def SD():
        return n_dim_sphere_init_data([0, 0, 0, 0, 0, 0], outer, batch_size)

    return f, XD, SD()


# 10-d version of pj_original
def eight_poly(batch_size, functions, inner=0., outer=10.):
    _And = functions['And']
    _Or = functions['Or']

    def f(_, v):
        x0, x1, x2, x3, x4, x5, x6, x7 = v
        # x^4 + 6 x^3 + 13 x^2 + 12 x + 4
        # is stable with roots in -0.5, -1, -2, -3
        return [x1, x2,
                x3,
                -6*x3 - 13*x2 - 12*x1 - 4*x0,
                x5, x6,
                x7,
                -6*x7 - 13*x6 - 12*x5 - 4*x4,
                ]

    def XD(_, v):
        x0, x1, x2, x3, x4, x5, x6, x7 = v
        return _And(inner**2 < x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2 + x7 ** 2,
                    x0 ** 2 + x1 ** 2 + x2 ** 2 + x3 ** 2 + x4 ** 2 + x5 ** 2 + x6 ** 2 + x7 ** 2 <= outer**2)

    def SD():
        return n_dim_sphere_init_data([0, 0, 0, 0, 0, 0, 0, 0], outer, batch_size)

    return f, XD, SD()


def benchmark_8(x):
    # todo: parametric model
    return [
        x[1],
        -(m+2)*x[0] - x[1]
    ]


def benchmark_9(x):
    # todo: parametric model
    return [
        x[1],
        -(m+2)*x[0] - x[1]
    ]


def max_degree_fx(fx):
    return max(max_degree_poly(f) for f in fx)


def max_degree_poly(p):
    s = str(p)
    s = re.sub(r'x\d+', 'x', s)
    try:
        f = sp.sympify(s)
        return sp.degree(f)
    except:
        print("Exception in %s for %s" % (max_degree_poly.__name__, p))
        return 0


if __name__ == '__main__':
    f, XD, SD = eight_poly(batch_size=500, functions={'And': z3.And, 'Or': None}, inner=0, outer=10.)
    plt.scatter(SD[:, 0].detach(), SD[:, 1].detach(), color='b')
    plt.show()