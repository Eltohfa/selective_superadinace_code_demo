import numpy as np
from numba import njit, prange, objmode
from math import erf, sqrt
from collections import namedtuple
from numba import types
from numba.typed import Dict

@njit
def cdf_normal(x, mu, sigma):
    return 0.5 * (1.0 + erf((x - mu) / (sigma * sqrt(2))))

@njit
def pi(i, j, N):
    if i >= j or i < 0 or j < 0 or i >= N or j >= N:
        raise ValueError("i and j must be in the range [0, N) and i < j")
    return (N * (N - 1)) // 2 - (N - i) * (N - i - 1) // 2 + j - i - 1


Params3 = namedtuple("Params", ["y", "N", "lut1", "lut2"])

@njit
def op_conj(op):
    ''' maps m (0) to  p (2), e (1) to itself, a (3) to ap (4) '''
    if 0 <= op <= 2:
        return 2 - op
    elif 3 <= op <= 4:
        return 7 - op
    else:
        raise ValueError("Invalid operator code")
    
@njit          
def E1(op,i,q):

    if op == 0:
        return q.y[q.lut1[0] + i]
    elif op == 1:
        return q.y[q.lut1[1] + i]
    elif op == 2:
        return np.conj(q.y[q.lut1[0] + i])
    else:
        raise ValueError("Invalid combination")
    
@njit
def E2(op1,i, op2,j,q):
    if not (0 <= i < j < q.N):
        if not (i<j):
            return E2(op2,j, op1,i,q)
        else:
            raise ValueError("Require 0 <= i < j < N")

    prefix = q.lut2[op1, op2]    

    if prefix != -1:
        return q.y[prefix + pi(i, j, q.N)]
    else:
        return np.conj(E2(op_conj(op1), i, op_conj(op2), j,q))


@njit    
def E3(op1,i, op2,j, op3, k,q):

    term1 = E2(op1,i, op2,j, q) * E1(op3, k, q)
    term2 = E2(op1,i, op3,k, q) * E1(op2, j, q)
    term3 = E2(op2,j, op3,k, q) * E1(op1, i, q)
    term4 = E1(op3, k, q) * E1(op2, j, q) * E1(op1, i, q)
    return term1 + term2 + term3 - 2 * term4


@njit
def interp_linear(t, tlist, flist):
    """
    Linear interpolation of f(t) given arrays tlist and flist.
    Assumes tlist is sorted and t is within [tlist[0], tlist[-1]].
    """
    n = len(tlist)

    # Handle out-of-bounds explicitly if needed
    if t <= tlist[0]:
        return flist[0]
    if t >= tlist[-1]:
        return flist[-1]

    # Binary search for interval
    left = 0
    right = n - 1
    while right - left > 1:
        mid = (left + right) // 2
        if tlist[mid] <= t:
            left = mid
        else:
            right = mid

    # Linear interpolation
    t0 = tlist[left]
    t1 = tlist[left + 1]
    f0 = flist[left]
    f1 = flist[left + 1]

    return f0 + (f1 - f0) * (t - t0) / (t1 - t0)




@njit#(parallel=True)
def rhs_mf2(t,y, h, H, N, lut, pulse_type = 'square', tf = 5/30.5):
    """mean field right hand side"""
    # where things stored in the vector
    size_two = N * (N - 1) // 2 # number of ordered pairs of a set of size N: (1,2), (1,3), ... (2,3), * (2,4), ...

    sys_size = 2 * N + 5 * size_two

    # where things stored in the vector
    idx_e = 0 # e
    idx_m = idx_e + N # sm
    idx_ee = idx_m + N # ee terms
    idx_em = idx_ee + size_two # e sm terms
    idx_me = idx_em + size_two # sm e terms
    idx_mm = idx_me + size_two # s- s- terms
    idx_pm = idx_mm + size_two # s+ s- terms

    # operator codes
    e = 1
    m = 0
    p = 2


    sys_size = len(y)
    lut1, lut2 = lut

    # with objmode(t0='f8'):
    #     t0 = time.perf_counter()
    q = Params3(y, N, lut1, lut2)
    # with objmode(t1='f8'):
    #     t1 = time.perf_counter()
    # print("Part A: setting p params time:", t1 - t0)        
    # input field
    # tr = -4.0/30.5 # rise time of the pulse
    # tf = 0.0/30.5 # falling of the pulse
    # tau = 0.1/30.5 # smoothness parameter 0.1 rise time

    tr = 0.0/30.5 # rise time of the pulse
    # tf = 5.0/30.5 # falling of the pulse
    tau = 0.1/30.5 # smoothness parameter 0.1 rise time

    if pulse_type == 'square':
        h = h * (t<tf) * (t>=tr)
    elif pulse_type == 'err':
        h = h * (cdf_normal(t, tr, tau) - cdf_normal(t, tf, tau))
    elif pulse_type == 'fr':
        tmid = 36**(1/6) * 0.07 / 2
        h = h * np.exp(-(2*(t-tmid)/0.07)**6)
    # calculate the auxiliary fields recursively

    dydt = np.zeros(sys_size, dtype=np.complex128)

    # dydt[:N] = - np.diag(H) * y[:N] - 1j * np.sqrt(beta_values) * (np.diag(pf) - np.conj(np.diag(pf)))

    # dydt[idx_m:idx_ee] = - y[idx_m:idx_ee]/2 + 1j  * np.sqrt(beta_values) * (-f + 2*np.diag(ef))

    for k in range(N):
        s = 0
        s1 = 0
        for i in range(N):
            if i != k:
                s += H[i,k] * E2(p, k, m, i, q)
                s1 += H[i,k] * (2 * E2(e, k, m, i, q) - E1(m, i, q))
        term = - (H[k,k] * y[k] + s + h[k] * E1(p, k, q))
        dydt[k] = term + np.conj(term)
        dydt[idx_m + k] = s1 + (2 * E1(e, k, q) - 1) * h[k] - H[k,k] * E1(m, k, q)

    # for MF1 type

    # for k in range(N):
    #     s = 0
    #     s1 = 0
    #     for i in range(N):
    #         if i != k:
    #             s += H[i,k] * E1(p, k, q) * E1(m, i, q)
    #             s1 += H[i,k] * (2 * E1(e, k, q) * E1(m, i, q) - E1(m, i, q))
    #     term = - (H[k,k] * y[k] + s + h[k] * E1(p, k, q))
    #     dydt[k] = term + np.conj(term)
    #     dydt[idx_m + k] = s1 + (2 * E1(e, k, q) - 1) * h[k] - H[k,k] * E1(m, k, q)

    # print("finished 1 site ops")
    # fill the ee, em, me, mm, pm terms using y and the auxiliary fields
    # ...
    idx_em = lut2[1, 0]

    # redefine the paramters to reflect the changes in the auxilary fields
    # q = Params3(y,f,ef, mf, pf, pef, eef, mmf, ppf, mef, pmf, N, lut1, lut2, lut3)
    # with objmode(t0='f8'):
    #     t0 = time.perf_counter()
    # for k in prange(N):

    for k in prange(N):
        for l in range(k):
            # lk indexes
            pmlk = idx_pm + pi(l, k, N)
            eelk = idx_ee + pi(l, k, N)
            emlk = idx_em + pi(l, k, N)
            melk = idx_me + pi(l, k, N)
            mmlk = idx_mm + pi(l, k, N)


            # pm term
             
            # term1 from deriv of ml term2 from deriv of pk
            # term2 is term1 with l exchanged with k and then conj is taken

            s1 = 0
            s2 = 0
            for i in range(N):
                if i != k and i != l:
                    s1 += H[i,l] * (2 * E3(p, k, e, l, m, i, q) - E2(p, k, m, i, q))
                    s2 += H[i,k] * (2 * E3(p, l, e, k, m, i, q) - E2(p, l, m, i, q))

            term1 = s1
            term1 += h[l] * (2 * E2(p, k, e, l, q) - E1(p, k, q))
            term1 += - H[l, l] * E2(p, k, m, l, q)
            term1 += H[k, l] * (2 * E2(e, l, e, k, q) - E1(e, k, q))

            term2 = s2
            term2 += h[k] * (2 * E2(p, l, e, k, q) - E1(p, l, q))
            term2 += - H[k, k] * E2(p, l, m, k, q)
            term2 += H[l, k] * (2 * E2(e, k, e, l, q) - E1(e, l, q))

            # dydt[pmlk] = term1 + np.conj(term2)
            dydt[pmlk] = np.conj(term1) + term2

            # mm term
            # term1 from deriv of ml term2 from deriv of mk
            # term2 is term1 with l exchanged with k

            s1 = 0
            s2 = 0
            for i in range(N):
                if i != k and i != l:
                    s1 += H[i,l] * (2 * E3(m, k, e, l, m, i, q) - E2(m, k, m, i, q))
                    s2 += H[i,k] * (2 * E3(m, l, e, k, m, i, q) - E2(m, l, m, i, q))

            term1 = s1
            term1 += h[l] * (2 * E2(m, k, e, l, q) - E1(m, k, q))
            term1 += - H[l, l] * E2(m, k, m, l, q)

            term2 = s2
            term2 += h[k] * (2 * E2(m, l, e, k, q) - E1(m, l, q))
            term2 += - H[k, k] * E2(m, l, m, k, q)

            dydt[mmlk] = term1 + term2

            # em term
            # term1 and term2 are from deriv of el, term3 is from deriv of mk

            s1 = 0
            s2 = 0
            s3 = 0
            for i in range(N):
                if i != k and i != l:
                    s1 += H[i,l] * E3(p, l, m, i, m, k, q)
                    s2 += np.conj(H[i, l]) * E3(m, l, p, i, m, k, q)
                    s3 += H[i,k] * (2*E3(e, l, e, k, m, i, q) - E2(e, l, m, i, q))

            term1 = H[l,l] * E2(e, l, m, k, q) + s1 + h[l] * E2(p, l, m, k, q)
            term2 = H[l,l] * E2(e, l, m, k, q) + s2 + np.conj(h[l]) * E2(m, l, m, k, q)
            term2 += np.conj(H[k,l]) * E2(m, l, e, k, q)
            term3 = s3 + h[k] * (2*E2(e, l, e, k, q) - E1(e, l, q)) - H[k,k] * E2(e, l, m, k, q)

            dydt[emlk] = - term1 - term2 + term3



            # me terms should be like em with k exchanged with l

            s1 = 0
            s2 = 0
            s3 = 0
            for i in range(N):
                if i != l and i != k:
                    s1 += H[i,k] * E3(p, k, m, i, m, l, q)
                    s2 += np.conj(H[i, k]) * E3(m, k, p, i, m, l, q)
                    s3 += H[i,l] * (2*E3(e, k, e, l, m, i, q) - E2(e, k, m, i, q))

            term1 = H[k,k] * E2(e, k, m, l, q) + s1 + h[k] * E2(p, k, m, l, q)
            term2 = H[k,k] * E2(e, k, m, l, q) + s2 + np.conj(h[k]) * E2(m, k, m, l, q)
            term2 += np.conj(H[l,k]) * E2(m, k, e, l, q)
            term3 = s3 + h[l] * (2*E2(e, k, e, l, q) - E1(e, k, q)) - H[l,l] * E2(e, k, m, l, q)

            dydt[melk] = - term1 - term2 + term3


            ## ee terms

            s1 = 0
            s2 = 0
            for i in range(N):
                if i != k and i != l:
                    s1 += H[i,l] * E3(p, l, m, i, e, k, q)
                    s2 += H[i,k] * E3(p, k, m, i, e, l, q)

            # double check the zero? these terms do not appear in the julia output equations
            term1 = H[l,l] * E2(e, l, e, k, q) + s1 + h[l] * E2(p, l, e, k, q) + 0 * H[k,l] * E2(p, l, m, k, q)
            term2 = H[k,k] * E2(e, k, e, l, q) + s2 + h[k] * E2(p, k, e, l, q) + 0 * H[l,k] * E2(p, k, m, l, q)

            dydt[eelk] = - (term1 + term2 + np.conj(term1 + term2))


    return dydt

@njit
def compute_powers(Gfs, Gr, sol_y, idx_m, idx_e, idx_pm, pi_func, N, Nt):
    """
    """

    # # Precompute sqrt(beta)
    # sqrt_beta = np.sqrt(beta_values)

    # # s_out (Nt,)
    # s_out = np.empty(Nt, dtype=np.complex128)
    # for l in range(Nt):
    #     total = 0.0j
    #     for n in range(N):
    #         total += sqrt_beta[n] * sol_y[idx_m + n, l]
    #     s_out[l] = total

    # # f_out (Nt,)
    # f_out = alpha_t - 1j * s_out

    # sdags base term: sum(beta * e->m)

    sdagsfs = np.empty(Nt, dtype=np.float64)
    for l in range(Nt):
        total = 0.0j
        for n in range(N):
            total += np.real(Gfs[n,n]) * sol_y[idx_e + n, l]
        sdagsfs[l] = total.real

    sdagswg = np.empty(Nt, dtype=np.float64)
    for l in range(Nt):
        total = 0.0j
        for n in range(N):
            total += np.real(Gr[n,n]) * sol_y[idx_e + n, l]
        sdagswg[l] = total.real

    # # add pair contributions (pmmn terms)
    # for l in range(Nt):
    #     add_term = 0.0
    #     for n in range(N):
    #         for m in range(n):
    #             pmmn = sol_y[idx_pm + pi_func(m, n, N), l]
    #             add_term += np.sqrt(beta_values[n] * beta_values[m]) * 2.0 * pmmn.real
    #     sdags[l] += add_term

    # add pair contributions (pmmn terms)
    for l in range(Nt):
        add_term = 0.0
        for n in range(N):
            for m in range(n):
                pmmn = sol_y[idx_pm + pi_func(m, n, N), l]
                add_term += np.real(Gfs[n,m]) * 2.0 * pmmn.real
        sdagsfs[l] += add_term

    for l in range(Nt):
        add_term = 0.0
        for n in range(N):
            for m in range(n):
                pmmn = sol_y[idx_pm + pi_func(m, n, N), l]
                add_term += np.real(Gr[n,m]) * 2.0 * pmmn.real
        sdagswg[l] += add_term

    # # incoherent power
    # p_inc_out = sdags - np.abs(s_out) ** 2

    # # coherent power
    # power_output_coherent = np.abs(f_out) ** 2

    p_fs = sdagsfs
    p_wg = sdagswg

    # return power_output_coherent, p_inc_out

    return p_fs, p_wg