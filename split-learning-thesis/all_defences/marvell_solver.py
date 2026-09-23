import math
import random

OBJECTIVE_EPSILON = 1e-16
CONVEX_EPSILON = 1e-20
NUM_CANDIDATE = 1


def symKL_objective(lam10, lam20, lam11, lam21, u, v, d, g):
    if (lam21 + v) == 0.0 or (lam20 + u) == 0.0 or (lam11 + v) == 0.0 or (lam10 + u) == 0.0:
        return float('inf')
    objective = (d - 1) * (lam20 + u) / (lam21 + v) \
                + (d - 1) * (lam21 + v) / (lam20 + u) \
                    + (lam10 + u + g) / (lam11 + v) \
                        + (lam11 + v + g) / (lam10 + u)
    return objective


def symKL_objective_zero_uv(lam10, lam11, g):
    objective = (lam10 + g) / lam11 \
                    + (lam11 + g) / lam10
    return objective


def solve_isotropic_covariance(u, v, d, g, p, P,
                               lam10_init=None, lam20_init=None,
                               lam11_init=None, lam21_init=None):
    if u == 0.0 and v == 0.0:
        return solve_zero_uv(g=g, p=p, P=P)

    ordering = [0, 1, 2]
    random.shuffle(x=ordering)

    solutions = []
    if u <= v:
        for i in range(NUM_CANDIDATE):
            if i % 3 == ordering[0]:
                if lam20_init:
                    lam20 = lam20_init

                else:
                    lam20 = random.random() * P / (1-p) / d
                lam10, lam11 = None, None

            elif i % 3 == ordering[1]:
                if lam11_init:
                    lam11 = lam11_init
                else:
                    lam11 = random.random() * P / p
                lam10, lam20 = None, None

            else:
                if lam10_init:
                    lam10 = lam10_init
                else:
                    lam10 = random.random() * P / (1-p)
                lam11, lam20 = None, None


            solutions.append(solve_small_neg(u=u,v=v,d=d,g=g,p=p,P=P, lam10=lam10, lam11=lam11, lam20=lam20))
        
    else:
        for i in range(NUM_CANDIDATE):
            if i % 3 == ordering[0]:
                if lam21_init:
                    lam21 = lam21_init
                else:
                    lam21 = random.random() * P / p / d
                lam10, lam11 = None, None

            elif i % 3 == ordering[1]:
                if lam11_init:
                    lam11 = lam11_init
                else:
                    lam11 = random.random() * P / p
                lam10, lam21 = None, None

            else:
                if lam10_init:
                    lam10 = lam10_init
                else:
                    lam10 = random.random() * P / (1-p)
                lam11, lam21 = None, None


            solutions.append(solve_small_pos(u=u,v=v,d=d,g=g,p=p,P=P, lam10=lam10, lam11=lam11, lam21=lam21))


    lam10, lam20, lam11, lam21, objective = min(solutions, key=lambda x: x[-1])


    return (lam10, lam20, lam11, lam21, objective)


def solve_zero_uv(g, p, P):
    C = P

    E = math.sqrt((C + (1 - p) * g) / (C + p * g))
    tau = max((P / p) / (E + (1 - p)/p), 0.0)

    if 0 <= tau and tau <= P / (1 - p):
        lam10 = tau
        lam11 = max(P / p - (1 - p) * tau / p, 0.0)
    else:
        lam10_case1, lam11_case1 = 0.0, max(P/p, 0.0)
        lam10_case2, lam11_case2 = max(P/(1-p), 0), 0.0
        objective1 = symKL_objective_zero_uv(lam10=lam10_case1,lam11=lam11_case1,
                                             g=g)
        objective2 = symKL_objective_zero_uv(lam10=lam10_case2,lam11=lam11_case2,
                                             g=g)
        if objective1 < objective2:
            lam10, lam11 = lam10_case1, lam11_case1
        else:
            lam10, lam11 = lam10_case2, lam11_case2
    
    objective = symKL_objective_zero_uv(lam10=lam10, lam11=lam11, g=g)

    return (lam10, 0.0, lam11, 0.0, 0.5 * objective - 1) 


def solve_small_neg(u, v, d, g, p, P, lam10=None, lam20=None, lam11=None):
    LAM21 = 0.0
    i = 0
    objective_value_list = []

    if lam20:
        ordering = [0, 1, 2]
    elif lam11:
        ordering = [1, 0, 2]
    else:
        ordering = [1, 2, 0]


    while True:
        if i % 3 == ordering[0]:
            D = P - (1 - p) * (d - 1) * lam20
            C = D + p * v + (1 - p) * u

            E = math.sqrt((C + (1 - p) * g) / (C + p * g))
            tau = max((D / p + v - E * u) / (E + (1 - p)/p), 0.0)

            if lam20 <= tau and tau <= P / (1 - p) - (d - 1) * lam20:
                lam10 = tau
                lam11 = max(D / p - (1 - p) * tau / p, 0.0)
            else:
                lam10_case1, lam11_case1 = lam20, max(P/p - (1-p)*d*lam20/p, 0.0)
                lam10_case2, lam11_case2 = max(P/(1-p) - (d-1)*lam20, 0), 0.0
                objective1 = symKL_objective(lam10=lam10_case1,lam20=lam20,lam11=lam11_case1,lam21=LAM21,
                                             u=u, v=v, d=d, g=g)
                objective2 = symKL_objective(lam10=lam10_case2,lam20=lam20,lam11=lam11_case2,lam21=LAM21,
                                             u=u, v=v, d=d, g=g)
                if objective1 < objective2:
                    lam10, lam11 = lam10_case1, lam11_case1
                else:
                    lam10, lam11 = lam10_case2, lam11_case2

        elif i % 3 == ordering[1]:
            D = max((P - p * lam11) / (1 - p), 0.0)
            f = lambda x: symKL_objective(lam10=D - (d-1)*x, lam20=x, lam11=lam11, lam21=LAM21,
                                          u=u, v=v, d=d, g=g)


            def f_prime(x):
                if x == 0.0 and u == 0.0:
                    return float('-inf')
                else:
                    return (d-1)/v - (d-1)/(lam11+v) - (d-1)/(x+u)*(v/(x+u)) + (lam11 + v + g)/(D-(d-1)*x+u) * ((d-1)/(D-(d-1)*x+u))


            lam20 = convex_min_1d(xl=0.0, xr=D/d, f=f, f_prime=f_prime)
            lam10 = max(D - (d-1) * lam20, 0.0)

        else:
            D = max(P - (1 - p) * lam10, 0.0)
            f = lambda x: symKL_objective(lam10=lam10, lam20=x, lam11=D/p - (1-p)*(d-1)*x/p, lam21=LAM21,
                                          u=u, v=v, d=d, g=g)


            def f_prime(x):
                if x == 0.0 and u == 0.0:
                    return float('-inf')
                else:
                    return (d-1)/v - (1-p)*(d-1)/(lam10 + u)/p - (d-1)/(x+u)*(v/(x+u)) + (lam10+u+g)/(D/p - (1-p)*(d-1)*x/p + v) * (1-p) * (d-1) / p / (D/p - (1-p)*(d-1)*x/p + v)


            lam20 = convex_min_1d(xl=0.0, xr=min(D/((1-p)*(d-1)), lam10), f=f, f_prime=f_prime)
            lam11 = max(D/p - (1-p)*(d-1)*lam20/p, 0.0)

        if lam10 <0 or lam20 < 0 or lam11 <0 or LAM21 <0:
            assert False, i

        objective_value_list.append(symKL_objective(lam10=lam10,lam20=lam20,lam11=lam11,lam21=LAM21,
                                             u=u, v=v, d=d, g=g))


        if (i>=3 and objective_value_list[-4] - objective_value_list[-1] < OBJECTIVE_EPSILON) or i >= 100:
            return lam10, lam20, lam11, LAM21, 0.5 * objective_value_list[-1] - d

        i += 1


def solve_small_pos(u, v, d, g, p, P, lam10=None, lam11=None, lam21=None):
    LAM20 = 0.0
    i = 0
    objective_value_list = []
    if lam21:
        ordering = [0, 1, 2]
    elif lam11:
        ordering = [1, 0, 2]
    else:
        ordering = [1, 2, 0]

    while True:
        if i % 3 == ordering[0]:
            D = P - p * (d - 1) * lam21
            C = D + p * v + (1 - p) * u

            E = math.sqrt((C + (1 - p) * g) / (C + p * g))
            tau = max((D / p + v - E * u) / (E + (1 - p)/p), 0.0)

            if 0.0 <= tau and tau <= (P - p*d*lam21)/(1-p):
                lam10 = tau
                lam11 = max(D / p - (1 - p) * tau / p, 0.0)
            else:
                lam10_case1, lam11_case1 = 0, max(P/p - (d-1)*lam21, 0.0)
                lam10_case2, lam11_case2 = max((P - p*d*lam21)/(1-p), 0.0), lam21
                objective1 = symKL_objective(lam10=lam10_case1,lam20=LAM20,lam11=lam11_case1,lam21=lam21,
                                             u=u, v=v, d=d, g=g)
                objective2 = symKL_objective(lam10=lam10_case2,lam20=LAM20,lam11=lam11_case2,lam21=lam21,
                                             u=u, v=v, d=d, g=g)
                if objective1 < objective2:
                    lam10, lam11 = lam10_case1, lam11_case1
                else:
                    lam10, lam11 = lam10_case2, lam11_case2

        elif i % 3 == ordering[1]:
            D = max(P - p * lam11, 0.0)
            f = lambda x: symKL_objective(lam10=(D - p*(d-1)*x)/(1-p), lam20=LAM20, lam11=lam11, lam21=x,
                                          u=u, v=v, d=d, g=g)


            def f_prime(x):
                if x == 0.0 and v == 0.0:
                    return float('-inf')
                else:
                    return (d-1)/u - p*(d-1)/(lam11+v)/(1-p) - (d-1)/(x+v)*(u/(x+v)) + (lam11 + v + g) / ((D - p*(d-1)*x)/(1-p) + u) * p * (d-1) / (1-p) /((D - p*(d-1)*x)/(1-p) + u)


            lam21 = convex_min_1d(xl=0.0, xr=min(D/p/(d-1), lam11), f=f, f_prime=f_prime)
            lam10 = max((D - p*(d-1)*lam21)/(1-p), 0.0)

        else:
            D = max((P - (1 - p) * lam10) / p, 0.0)
            f = lambda x: symKL_objective(lam10=lam10, lam20=LAM20, lam11=D - (d-1)*x, lam21=x,
                                          u=u, v=v, d=d, g=g)


            def f_prime(x):
                if x == 0.0 and v == 0.0:
                    return float('-inf')
                else:
                    return (d-1)/u - (d-1)/(lam10+u) - (d-1)/(x+v)*(u/(x+v)) + (lam10 + u + g)/(D-(d-1)*x+v) * (d-1) / (D-(d-1)*x+v)


            lam21 = convex_min_1d(xl=0.0, xr=D/d, f=f, f_prime=f_prime)
            lam11 = max(D - (d-1) * lam21, 0.0)

        if lam10 <0 or LAM20 <0 or lam11 <0 or lam21 <0:
            assert False, i

        objective_value_list.append(symKL_objective(lam10=lam10,lam20=LAM20,lam11=lam11,lam21=lam21,
                                             u=u, v=v, d=d, g=g))


        if (i>=3 and objective_value_list[-4] - objective_value_list[-1] < OBJECTIVE_EPSILON) or i >= 100:
            return lam10, LAM20, lam11, lam21, 0.5 * objective_value_list[-1] - d

        i += 1


def convex_min_1d(xl, xr, f, f_prime):
    assert xr <= 1e5
    assert xl <= xr, (xl, xr)


    xm = (xl + xr) / 2


    if abs(xl - xr) <= CONVEX_EPSILON:
        return min((f(x), x) for x in [xl, xm, xr])[1]
    if f_prime(xl) <=0 and f_prime(xr) <= 0:
        return xr
    elif f_prime(xl) >=0 and f_prime(xr) >= 0:
        return xl
    if f_prime(xm) > 0:
        return convex_min_1d(xl=xl, xr=xm, f=f, f_prime=f_prime)
    else:
        return convex_min_1d(xl=xm, xr=xr, f=f, f_prime=f_prime)


def small_neg_problem_string(u, v, d, g, p, P):
    return 'minimize ({2}-1)*(z + {0})/{1} + ({2}-1)*{1}/(z+{0})+(x+{0}+{3})/(y+{1}) + (y+{1}+{3})/(x+{0}) subject to x>=0, y>=0, z>=0, z<=x, {4}*y+(1-{4})*x+(1-{4})*({2}-1)*z={5}'.format(u,v,d,g,p,P)

def small_pos_problem_string(u, v, d, g, p, P):
    return 'minimize ({2}-1)*{0}/(z+{1}) + ({2}-1)*(z + {1})/{0} + (x+{0}+{3})/(y+{1}) + (y+{1}+{3})/(x+{0}) subject to x>=0, y>=0, z>=0, z<=y, {4}*y+(1-{4})*x+{4}*({2}-1)*z={5}'.format(u,v,d,g,p,P)

def zero_uv_problem_string(g, p, P):
    return 'minimize (x+{0})/y + (y+{0})/x subject to x>=0, y>=0, {1}*y+(1-{1})*x={2}'.format(g,p,P)

if __name__ == '__main__':
    import random
    import time
    from collections import Counter

    test_neg = False


    u = 3.229033590534426e-15
    v = 3.0662190349955726e-15
    d = 128.0
    g = 5.015613264502392e-10
    p = 0.253936767578125
    P = 2328365.0213796967

    print('u={0},v={1},d={2},g={3},p={4},P={5}'.format(u,v,d,g,p,P))
    start = time.time()
    lam10, lam20, lam11, lam21, sumKL = solve_isotropic_covariance(u=u, v=v, d=d, g=g, p=p, P=P)
    print(lam10, lam20, lam11, lam21, sumKL)
    print('time', time.time() - start)
    if u < v:
        print(small_neg_problem_string(u=u,v=v,d=d,g=g,p=p,P=P))
    else:
        print(small_pos_problem_string(u=u,v=v,d=d,g=g,p=p,P=P))

    start = time.time()
    print(solve_isotropic_covariance(u=u, v=v, d=d, g=g, p=p, P=P + 10, 
                                     lam10_init=lam10, lam20_init=lam20,
                                     lam11_init=lam11, lam21_init=lam21))
    print('time', time.time() - start)