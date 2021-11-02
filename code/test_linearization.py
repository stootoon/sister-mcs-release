# Run all tests with 'python -m unittest test_linearization'
# Run a speciic test with 'python -m unittest test_linearization.SparseOdour.test_v0_oscillating_modes'
import os, sys
import unittest
import logging
from tqdm import tqdm

import numpy as np
from numpy import *

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import sympy
from sympy import symbols, expand, init_printing, Poly, simplify, solve, Eq
is_sympy = lambda x: "sympy" in str(type(x))
from cmath import sqrt as csqrt
from collections import namedtuple

sys.path.append(os.path.join(os.environ["SISTER_MCS"], "code"))
import olfactory_bulb
import datatools as dt
import util

import pdb

logger = util.create_logger("test_linearization")
INFO   = logger.info

Eigen = namedtuple('Eigen', ['val', 'vec'])
def check_eigen(P, eigen, **kwargs):
    return allclose(P @ eigen.vec, eigen.val * eigen.vec, **kwargs)

def find_closest(needle, haystack, return_index = False):
    inearest = argmin(abs(needle - haystack))
    return haystack[inearest] if return_index is False else (haystack[inearest], inearest)

absolute_distance = lambda x, nearest: abs(x - nearest)
relative_distance = lambda x, nearest: abs(x - nearest)/abs(nearest)
symmetric_relative_distance = lambda x, y: abs(x - y)/min(abs(x),abs(y))

def distance_to_points(needles, haystack, dist_fun = absolute_distance):
    """For each needles, finds the distance to the nearest memeber of
    haystack.

    Returns: The array of distances from each needle to the haystack.
    """
    dists = array([dist_fun(needle, find_closest(needle, haystack)) for needle in needles])
    return dists

def approximate_subset(needles, haystack, dist_fun, threshold):
    """Tests whether all elements of needles are approximate subsets
    of haystack. Approximate membership means distance to nearest
    point haystack is less than threshold.

    Returns: A tuple containing:
    - A boolean flag indicating that all needles were within the
    specified threshold distance of the haystack;

    - the array of distances from each needle to the haystack. 
    """
    dists = distance_to_points(needles, haystack, dist_fun)
    return all(dists<=threshold), dists

def _test_approximate_equality(self, a_name, a, b_name, b, tol = 1e-3, dist_fun = symmetric_relative_distance):
    """Tests whether a ≈ b by checking whether their distance as
    computed by dist_fun is less than the specified tolerance.
    """    
    dist = dist_fun(a, b)
    INFO(f"dist_fun({a_name}, {b_name}) = {dist:1.3e}.")    
    self.assertEqual(dist<tol, True)
    INFO(f"{a_name} is equal to {b_name} within tolerance {tol:1.3e}.")
    INFO(f"abs({a_name} - {b_name}) = {abs(a-b):1.3e}.")
            
def _test_approximate_subset(self, needles_name, needles, haystack_name, haystack, pc_tol = 1, plot=False):
    """Tests whether needles are an approximate subset of haystack by
    checking that all needles are within the specified percentage
    relative distance to the haystack. Setting 'plot' to True will
    generate a plot of the points in needles and haystack.
    """
    
    is_approximate_subset, dists = approximate_subset(needles, haystack, dist_fun = relative_distance, threshold=pc_tol/100)
    
    match_inds      = [argmin(dists), argmax(dists)]
    match_needles   = [needles[i] for i in match_inds]
    match_haystacks = [find_closest(needle, haystack) for needle in match_needles]

    if plot is True:
        INFO(f"Plotting {needles_name} and {haystack_name}.")
        ttl = f"Comparing {needles_name} to {haystack_name}"
        plt.figure(figsize=(11,8))
        plt.plot(real(haystack), imag(haystack), "go", label=f"haystack: {haystack_name}", fillstyle="none")
        plt.plot(real(needles), imag(needles),   "rx", label=f"needles: {needles_name}")
        for i, (lab, style) in enumerate([("closest", ":"), ("farthest", "--")]):
            plt.plot([real(match_needles[i]), real(match_haystacks[i])],
                     [imag(match_needles[i]), imag(match_haystacks[i])],
                     color="black",linestyle=style, label=lab)
        plt.xlabel("real part")
        plt.ylabel("imaginary part")
        plt.title(ttl)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        file_name = ttl.replace(" ","_")+".pdf"
        plt.savefig(file_name, bbox_inches="tight")
        INFO(f"Wrote plot to {file_name}.")
            
    error_pc = dists * 100
    INFO(f"Mean +/- std of error percentages: {mean(error_pc):1.3f} +/- {std(error_pc):1.3f}.")
    INFO(f" Closest:  {match_needles[0]:<16.3f} to {match_haystacks[0]:>16.3f} ({min(error_pc):>6.2f}%)")
    INFO(f"Farthest:  {match_needles[1]:<16.3f} to {match_haystacks[1]:>16.3f} ({max(error_pc):>6.2f}%)")    
    
    self.assertEqual(is_approximate_subset, True)
    
    INFO(f"All {len(needles)} {needles_name} were within {pc_tol}% of an element of {haystack_name}.")

def compute_predicted_eigenvalues(self, as_test = False):
    """Assuming 'self' is an object of type OlfactoryBulb for which
    the linearization matrix H has been computed, computes the
    predicted eigenvalues and their counts for the static and v = 0
    cases. 

    Returns: 
    - A dict of whose keys indicate the type of eigenvalue and whose
    values contain as many copies of those eigenvalues as predicted.

    - The remaining eigenvalues of H not accounted for above.
    """

    obs_eigs = np.linalg.eigvals(self.H)
    
    pred_eigs = {}
    pred_eigs["static"] = self.M  * [-self.leak/self.tau_mu]
    if self.n < self.M:
        pred_eigs["v0_decaying"] = (self.M - self.n) * [-1/self.tau_la]
    oscill_real = -(self.tau_mu + self.leak * self.tau_la)/(2*self.tau_mu * self.tau_la)
    oscill_sqrt = (self.tau_mu - self.leak*self.tau_la)**2/4/self.tau_la**2/self.tau_mu**2 - self.S * self.c/self.tau_la/self.tau_mu
    pred_eigs["v0_oscillating"] =  (self.M * (self.S - 1) - self.n) * [oscill_real + csqrt(oscill_sqrt)] 
    pred_eigs["v0_oscillating"]+=  (self.M * (self.S - 1) - self.n) * [oscill_real - csqrt(oscill_sqrt)]

    obs_eigs_left = copy(obs_eigs)
    tol = 1e-6
    for grp, grp_eigvals in pred_eigs.items():
        found = []
        for i, ev in enumerate(grp_eigvals):
            dist = abs(ev - obs_eigs_left)
            imin = argmin(dist)
            self.assertAlmostEqual(dist[imin],0,delta=tol)
            found.append([i])
            obs_eigs_left = np.delete(obs_eigs_left, imin)
        self.assertEqual(len(found), len(grp_eigvals))
        if as_test:
            status = "OK" if len(found) == len(grp_eigvals) else "ERROR"                
            INFO(f"{grp:>12s}: {len(found):>4d} / {len(grp_eigvals):<4d} found. {status}")
    return pred_eigs, obs_eigs_left

def _basic_setup(self, n, S = 8, leak = 0, partitioned = True, **kwargs):
    """Setup function for the unittests.
    """
    self.p = dt.load_default_params()
    self.p.update(**kwargs)
    self.p["k"]       = n
    self.p["leak_pg"] = leak
    self.p["S"]       = S
    self.leak   = self.p["leak_pg"] 
    self.tau_la = self.p["tau_mc"]
    self.tau_mu = self.p["tau_pg"]
    self.tau_v  = self.p["tau_gc"]
    self.M, self.N, self.S = [self.p[fld] for fld in "MNS"]
    self.ga  = self.p["ga"]
    self.c   = 1/self.p["sd"]**2
    self.eta = self.c/self.ga
    self.n   = self.p["k"]
    self.nu  = n / self.M if n > self.M else 1
    INFO(f"{self.n=} non-zero gcs.")

    # Use the partitioned matrix to ensure full rank W
    self.p["connectivity"] = "partitioned" if partitioned else "random"
    self.ob = olfactory_bulb.OlfactoryBulb(**self.p)

    self.active_gcs = where(olfactory_bulb.get_x_true(self.p["N"], self.n))[0]
    self.H = self.ob.linearize(self.active_gcs)

    self.al    = self.ob.A[:, self.active_gcs] 
    sum_al_alT = self.al.T @ self.al
    sum_WT_W   = zeros((self.n,self.n))
    for WTi in self.ob.W[:,self.active_gcs,:]:
        sum_WT_W += WTi @ WTi.T    

    _setup_equations(self)
        
def _setup_equations(self):
    # Eqn v-eigen
    lhs = lambda xi: (self.tau_v * xi + 1) * np.eye(self.n)
    rhs_coef_num = lambda xi: -self.eta * self.S * (self.tau_mu * xi + self.leak)
    rhs_coef_den = lambda xi: (self.tau_mu * xi + self.leak) * (self.tau_la * xi + 1) + self.ga * self.eta * self.S
    rhs_coef     = lambda xi: rhs_coef_num(xi) / rhs_coef_den(xi)
    rhs = lambda xi: rhs_coef(xi) * \
        (self.eta * self.ga/((self.tau_mu * xi + self.leak) * (self.tau_la * xi + 1)) * sum_al_alT + sum_WT_W)

    self.v_eigen = lambda xi: lhs(xi) - rhs(xi)
        
    # SYMBOLIC COMPUTATIONS
    # Trailing underscore will mean symbolic objects.
    tau_la_, tau_mu_, tau_v_ = symbols("tau_la_ tau_mu_ tau_v_")
    S_, c_, eta_, ga_, xi_   = symbols("S_ c_ eta_ gamma_ xi_")
    eps_, nu_, kmu_, kv_     = symbols("eps_ nu_ kmu_ kv_")

    # Substitution of numerical values for symbolics
    self.subs = [(tau_la_, self.tau_la), (tau_mu_, self.tau_mu), (tau_v_,  self.tau_v), (S_, self.S),
                 (c_, self.c), (eta_, self.eta), (ga_, self.ga),
                 (nu_, self.nu), (eps_, self.leak),
                 (kmu_, self.tau_la / self.tau_mu),
                 (kv_,  self.tau_la / self.tau_v)]        
    
    self.expr_2coefs_ = expr_2coefs_  = lambda expr: Poly(expr, xi_).coeffs()
    self.expr_2num    = expr_2num     = lambda expr,  astype=complex: astype(expr.subs(self.subs).evalf())
    self.list_2num    = list_2num     = lambda list_, astype=float:   array([item_.subs(self.subs).evalf() for item_ in list_]).astype(astype)
    self.coefs_2roots = coefs_2roots  = lambda coefs_: np.roots(list_2num(coefs_))
    self.expr_2roots  = expr_2roots   = lambda expr_:  coefs_2roots(expr_2coefs_(expr_))

    # q(ξ): Eqn qxi
    self.qxi_ =  (tau_la_**2 * tau_mu_ *  tau_v_) * xi_**4 + \
        (tau_la_**2 * tau_mu_ + 2 * tau_la_ * tau_mu_ * tau_v_ + eps_ * tau_la_**2 * tau_v_) * xi_**3  +\
    (ga_ * eta_ * S_ * tau_la_ * tau_v_ + eta_ * S_ * tau_la_ * tau_mu_ +\
     2 * tau_la_ * tau_mu_ + tau_mu_ * tau_v_ + eps_ * tau_la_**2 + 2 * eps_ * tau_la_ * tau_v_) * xi_**2 + \
     (ga_ * eta_ * S_ * tau_la_ + ga_ * eta_ * S_ * tau_v_ +  eta_ * S_ * tau_mu_ +\
      eta_ * S_ * eps_ * tau_la_ + tau_mu_ + eps_ * tau_v_ + 2 * eps_ * tau_la_ ) * xi_ + \
      nu_ * ga_ * eta_**2 * S_ + ga_ * eta_ * S_ + eps_ * eta_ * S_ + eps_

    self.qxi_coefs_ = expr_2coefs_(self.qxi_)        # Symbolic coefficients of q(ξ)
    self.qxi_roots  = coefs_2roots(self.qxi_coefs_)  # Numerical roots of q(ξ)

    # q0(ξ) is q(xi) but with nu set to 0, and the (tau_la ξ + 1) factored out.
    self.q0xi_ = (eps_ + S_ * ga_ * eta_ + S_ * eps_ * eta_) + \
        (eps_ * tau_v_ + S_ * ga_ * eta_ * tau_v_ + eps_ * tau_la_ + tau_mu_ + S_ * eta_ * tau_mu_) * xi_ +\
        (eps_ * tau_v_ * tau_la_ + tau_v_ * tau_mu_ + tau_la_ * tau_mu_) * xi_**2 +\
        (tau_v_ * tau_la_ * tau_mu_) * xi_**3

    self.q0xi_coefs_ = expr_2coefs_(self.q0xi_)
    self.q0xi_roots  = coefs_2roots(self.q0xi_coefs_)
    
    
def _test_exact_modes(self, which_mode):
    """ In this test we're checking that the predicted exact 
    modes are actual eigenvalues of the linearized system.

    Equations in the manuscript: static_eigs, decay_eigs, oscill_eigs
    """
    INFO(f"{self.name}._test_exact_modes({which_mode})")
    pred_eigs, _ = compute_predicted_eigenvalues(self)
    actual_eigs  = np.linalg.eigvals(self.H)
    
    for eigen in pred_eigs[which_mode]:
        closest, iclosest = find_closest(eigen, actual_eigs, return_index = True)
        self.assertTrue(np.allclose(eigen, closest))
        actual_eigs = np.delete(actual_eigs, iclosest)
    INFO(f"All {len(pred_eigs[which_mode])} {which_mode} eigenvalues found.")
    

def _test_det_v_eigen_zeros(self):
    """ In this test we're checking that all remaining eigenvalues
    (whose modes have v ≠ 0) are zeros of det(v_eigen(ξ)).
    Equations in the manuscript: v-eigen
    """
    INFO(f"{self.name}._test_det_v_eigen_zeros")
    pred_eigs, obs_eigs_left = compute_predicted_eigenvalues(self, as_test = True)
    # Check that the unaccounted for eigenvalues of P
    # are values of xi at which det(v_eigen) is zero.
    # We don't actually compute the determinant, because
    # we run into numerical issues. Instead we:
    # 1. Compute the minimum eigenvalue at xi and check that
    #    it's approximately zero.
    # 2. Compute the minimum eigenvalue at a nearby point and
    #    check that it's much larger.
    abs_tol = 1e-6
    rel_tol = 1e-6

    for ie, ee in enumerate(obs_eigs_left):
        r,i = real(ee), imag(ee)
        min_eig_self = np.min(abs(linalg.eigvals(self.v_eigen(r      + 1j*i)))) # Value of Qxi at xi
        min_eig_nbr  = np.min(abs(linalg.eigvals(self.v_eigen(r*0.99 + 1j*i)))) # Value at a neighbouring position
        rel_val      = min_eig_self/min_eig_nbr
        self.assertAlmostEqual(min_eig_self, 0, delta=abs_tol)
        self.assertAlmostEqual(rel_val,      0, delta=rel_tol)
    INFO(f"All {len(obs_eigs_left)} remaining eigenvalues were zeros of det(v_eigen(ξ)).")

class SparseOdour(unittest.TestCase):
    
    def setUp(self):
        INFO("SPARSE ODOUR TEST STARTED")
        np.random.seed(4)
        _basic_setup(self, n = 3, leak = 0.5)
        self.name = "SparseOdour"

    def tearDown(self):
        INFO("SPARSE ODOUR TEST ENDED")
        print("")

    def test_static_modes(self):
        _test_exact_modes(self, "static")

    def test_v0_decaying_modes(self):
        if self.n < self.M:
            _test_exact_modes(self, "v0_decaying")
        else:
            INFO("n >= M, so no decaying modes.")
            return True

    def test_v0_oscillating_modes(self): 
        _test_exact_modes(self, "v0_oscillating")

    def test_det_v_eigen_zeros(self):
        _test_det_v_eigen_zeros(self)

    def test_qxi_roots_vs_det_v_eigen_zeros(self):
        """ In this test we're checking that the roots of the quartic
        approximate q(ξ) to v_eigen(ξ) are within some percent of a zero of
        the determinant of v_eigen(ξ) and vise-versa.
        Equations in the manuscript: qxi, v-eigen """
        INFO(f"{self.name}.test_qxi_roots_vs_det_v_eigen_zeros")
        
        _, v_eigen_roots = compute_predicted_eigenvalues(self, as_test = False)        

        _test_approximate_subset(self, "roots of q(ξ)", self.qxi_roots, "zeros of det(v_eigen(ξ))", v_eigen_roots, pc_tol = 10,  plot = True)
        _test_approximate_subset(self, "zeros of det(v_eigen(ξ))", v_eigen_roots, "roots of q(ξ)", self.qxi_roots, pc_tol = 25,  plot = True)    
                    
    def test_qxi_roots_approx(self, run_test = True):
        """ In this test we want to check that expressions in 
        Eqn. high_low are a good approximation to the roots qxi. We 
        do this by checking that the roots are within some percentage 
        of each other. 
        
        Equations in the manuscript: qxi, high_low"""
        INFO(f"{self.name}.test_qxi_roots_approx")

        approx = self.ob.approximate_sparse_odour_eigs(nu = 1)
        qxi_low_roots_approx, qxi_high_roots_approx = [approx[fld] for fld in ["low", "high"]]
        
        qxi_low_roots  = array(sorted(self.qxi_roots, key=abs)[:2]) # Get the low roots of q(ξ)
        qxi_high_roots = array(sorted(self.qxi_roots, key=abs)[2:]) # Get the high roots of q(ξ)

        if run_test:
            _test_approximate_subset(self, "qxi_low_roots_approx", qxi_low_roots_approx, "low roots of q(ξ)",    qxi_low_roots,        pc_tol = 1, plot = True)
            _test_approximate_subset(self, "low roots of q(ξ)",    qxi_low_roots,        "qxi_low_roots_approx", qxi_low_roots_approx, pc_tol = 1, plot = True)        

            _test_approximate_subset(self, "qxi_high_roots_approx", qxi_high_roots_approx, "high roots of q(ξ)",    qxi_high_roots,        pc_tol = 1, plot = True)
            _test_approximate_subset(self, "high roots of q(ξ)",    qxi_high_roots,        "qxi_high_roots_approx", qxi_high_roots_approx, pc_tol = 1, plot = True)

        return {"qxi_low_roots":qxi_low_roots,
                "qxi_low_roots_approx":qxi_low_roots_approx,
                "qxi_high_roots":qxi_high_roots,
                "qxi_high_roots_approx":qxi_high_roots_approx}

class DenseOdour(unittest.TestCase):
    # The dense odour case is a bit more complicated because
    # the nu parameter is 0 for n - M, and n/M (on average)
    # for M.
    #
    # When nu = 0, q(xi) = (tau_la xi + 1) q0(xi).
    # q0(xi) is cubic, and (tau_la xi + 1) doesn't actually
    # contribute any roots to the system.
    #
    # So we have to check that the roots of the cubic q0(xi)
    # are approximate eigenvalues of the system.
    # And that q(xi), with nu as set as above,
    # are approximate eigenvalues of the system.
    
    def setUp(self):
        INFO("DENSE ODOUR TEST STARTED")
        np.random.seed(4)
        _basic_setup(self, n = 55, leak = 0.5)        
        self.name = "DenseOdour"

    def tearDown(self):
        INFO("DENSE ODOUR TEST ENDED")
        print("")

    def test_static_modes(self):
        _test_exact_modes(self, "static")

    def test_v0_oscillating_modes(self): 
        _test_exact_modes(self, "v0_oscillating")

    def test_det_v_eigen_zeros(self):
        _test_det_v_eigen_zeros(self)

    def test_qxi_roots_vs_det_v_eigen_zeros(self):
        """ In this test we're checking that the roots of the quartic
        approximate q(ξ) to v_eigen(ξ) are within some percent of a zero of
        the determinant of v_eigen(ξ). In the dense odour setting, only some 
        of the eigenvalues are contributed by q(xi), the rest are contributed
        by q0(ξ), which is q(ξ) but with nu set to 0 and (tau_la ξ + 1) 
        factored out. 

        Equations in the manuscript: qxi, v-eigen """
        INFO(f"{self.name}.test_qxi_roots_vs_det_v_eigen_zeros")
        
        _, v_eigen_roots = compute_predicted_eigenvalues(self, as_test = False)        
        _test_approximate_subset(self, "roots of q(ξ)",  self.qxi_roots,  "zeros of det(v_eigen(ξ))", v_eigen_roots, pc_tol = 10,  plot = True)
        _test_approximate_subset(self, "roots of q0(ξ)", self.q0xi_roots, "zeros of det(v_eigen(ξ))", v_eigen_roots, pc_tol = 10,  plot = True)        
                    
    def test_qxi_roots_approx(self, run_test = True):
        """ In this test we want to check that expressions in 
        Eqn. high_low are a good approximation to the roots qxi. We 
        do this by checking that the roots are within some percentage 
        of each other. 
        
        Equations in the manuscript: qxi, high_low"""
        INFO(f"{self.name}.test_qxi_roots_approx")

        approx = self.ob.approximate_dense_odour_eigs(self.nu)
        qxi_high_roots_approx, qxi_low_roots_approx, q0xi_high_roots_approx, q0xi_low_root_approx = \
            [approx[fld] for fld in ["qxi_high_roots_approx", "qxi_low_roots_approx", "q0xi_high_roots_approx", "q0xi_low_root_approx"]]
        
        qxi_low_roots  = np.array(sorted(self.qxi_roots, key=abs)[:2]) # Get the low roots of q(ξ)
        qxi_high_roots = np.array(sorted(self.qxi_roots, key=abs)[2:]) # Get the high roots of q(ξ)

        if run_test:
            _test_approximate_subset(self, "qxi_low_roots_approx",  qxi_low_roots_approx,  "low roots of q(ξ)",     qxi_low_roots,        pc_tol = 1, plot = True)
            _test_approximate_subset(self, "low roots of q(ξ)",     qxi_low_roots,         "qxi_low_roots_approx",  qxi_low_roots_approx, pc_tol = 1, plot = True)        

            _test_approximate_subset(self, "qxi_high_roots_approx", qxi_high_roots_approx, "high roots of q(ξ)",    qxi_high_roots,        pc_tol = 1, plot = True)
            _test_approximate_subset(self, "high roots of q(ξ)",    qxi_high_roots,        "qxi_high_roots_approx", qxi_high_roots_approx, pc_tol = 1, plot = True)

        q0xi_low_root   = np.array(sorted(self.q0xi_roots, key=abs)[:1]) # Get the low root of q0(ξ)
        q0xi_high_roots = np.array(sorted(self.q0xi_roots, key=abs)[1:]) # Get the high roots of q0(ξ)

        if run_test:
            _test_approximate_subset(self, "q0xi_low_root_approx",   q0xi_low_root_approx,   "low root of q0(ξ)",      q0xi_low_root,          pc_tol = 5, plot = True)
            _test_approximate_subset(self, "low root of q0(ξ)",      q0xi_low_root,          "q0xi_low_root_approx",   q0xi_low_root_approx,   pc_tol = 5, plot = True)        

            _test_approximate_subset(self, "q0xi_high_roots_approx", q0xi_high_roots_approx, "high roots of q0(ξ)",    q0xi_high_roots,        pc_tol = 5, plot = True)
            _test_approximate_subset(self, "high roots of q0(ξ)",    q0xi_high_roots,        "q0xi_high_roots_approx", q0xi_high_roots_approx, pc_tol = 5, plot = True)
            

        return {"qxi_low_roots":qxi_low_roots,
                "qxi_low_roots_approx":qxi_low_roots_approx,
                "qxi_high_roots":qxi_high_roots,
                "qxi_high_roots_approx":qxi_high_roots_approx,
                "q0xi_low_root":q0xi_low_root,
                "q0xi_low_root_approx":q0xi_low_root_approx,
                "q0xi_high_roots":q0xi_high_roots,
                "q0xi_high_roots_approx":q0xi_high_roots_approx,
        }
    
        
if __name__ == '__main__': 
    unittest.main()
