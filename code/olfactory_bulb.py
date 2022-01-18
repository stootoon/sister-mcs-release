import numpy as np
from cmath import sqrt as csqrt
import logging, json
import time, datetime
import cvxpy as cvx
import util

logger = util.create_logger("olfactory_bulb")
INFO   = logger.info

class TimedBlock:
    def __init__(self, name = "BLOCK"):
        self.name = name
        self.start_time = -1
        self.end_time = -1
        self.elapsed = -1

    def __enter__(self):
        self.start_time = time.time()
        INFO("Started  {}.".format(self.name))

    def __exit__(self, *args):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        INFO("Finished {} in {:.3f} seconds.".format(self.name, self.elapsed))

def linearly_rectify(x,th):
    return (x > th)*(x - th)

def smooth_rectify(x,th,f=10):
    return np.log(1 + np.exp(f*(x - th)))/f

def orth_extend_rows(P):
    # If the number of rows is less than the number of columns
    # Returns a remaining number of rows that are orthogonal to P
    # and to each other. Otherwise returns the empty matrix.
    Q = []
    nrows, ncols = P.shape
    for i in range(ncols-nrows):
        r = np.random.randn(ncols,)
        # Remove the component along P        
        # r = cP -> rP' = cPP' -> c = rP'(PP')^{-1} => cP = rP'(PP')^{-1}P

        Pr = np.linalg.multi_dot([r, P.T, np.linalg.inv(np.dot(P,P.T)), P])
        r -= Pr
        # Remove the components in Q.
        # These should be orthogonal to each other and to P.
        for q in Q:
            r -= np.dot(q,r)*q
        Q.append(r/np.linalg.norm(r))
            
    return np.array(Q)

def get_x_true(N, k, spread = 0):
    vals     = np.linspace(1-spread/2, 1+spread/2, k)
    ind_x_on = {int(round(i*float(N)/(k+1))) : vals[i-1] for i in range(1,k+1)}
    x_true   = np.zeros((N,))
    for ind, val in ind_x_on.items():
        x_true[ind] = val
        
    return x_true

class Odor:
    def __init__(self, x, A, amp = 1, plume={"shape":"step"}, t_on = 0, t_off = np.inf):
        self.x    = x
        self.A = A
        self.val0 = np.dot(A, x) if A is not None else x
        self.amp = amp
        self.plume = plume
        self.t_on = t_on
        self.t_off = t_off

        if plume["shape"] == "step":
            self.amp_fun = lambda t: amp * (t>=t_on)*(t<t_off)
        elif plume["shape"] == "sin":
            freq, phase, bias = [plume[fld] for fld in ["freq", "phase", "bias"]]
            self.amp_fun = lambda t: (t>=t_on)*(t<t_off) * (amp * np.sin(2 * np.pi * freq * t + phase) + bias)
        else:
            raise ValueError("Don't know what to do for plume shape={}.".format(p["shape"]))
    
        INFO(f"Odour Parameters:")
        INFO(f"  Plume from {t_on=} to {t_off=}.")
        INFO(f"  Using {amp=} and {plume}.")                    

    def value_at(self, t, A = None, noise_sd = 0):
        amp  = self.amp_fun(t)

        if not hasattr(t, "len") or len(t) == 1:
            val = amp * self.val0
        elif len(t) > 1:
            val = np.outer(amp, self.val0)
        else:
            raise ValueError("Length of time vector must be greater than 0, was {}".format(len(t)))

        if noise_sd:
            val += np.random.randn(*val.shape)*noise_sd

        return val

class OlfactoryBulb:

    def __init__(self, M=50, N=1000, S=8,
                 sd=1e-2, be=100, ga=100,
                 tau_mc=0.025, tau_gc=0.050, tau_pg=0.250,
                 leak_pg=0, active_gcs = "all",
                 noisy=False, verbosity=0, connectivity = "random", seed = 0, **kwargs):

        self.seed = seed
        np.random.seed(self.seed)
        
        self.M  = M
        self.N  = N
        self.S  = S
        self.sd = sd
        self.be = be
        self.ga = ga
        self.tau_mc  = tau_mc
        self.tau_gc  = tau_gc
        self.tau_pg  = tau_pg
        self.leak_pg = leak_pg
        self.A       = np.random.randn(M,N)/np.sqrt(M)
        self.noisy   = noisy
        self.verbosity = verbosity
        self.active_gcs = active_gcs
        self.connectivity = connectivity
        
        self.create_partitioned_sister_connectivity() if connectivity == "partitioned" else self.create_random_sister_connectivity()
        self.set_sister_cell_parameters()

        if self.verbosity>0:
            INFO("Created new olfactory bulb.")
            self.print_params()

    def print_params(self):
        INFO("OB Parameters:")
        INFO("  seed = %d." % self.seed)
        INFO("  M = %d, N = %d, S = %d." % (self.M,  self.N, self.S))
        INFO("  Using {} connectivity.".format(self.connectivity.upper()))
        INFO("  First 3 values of A: {}".format(self.A[0][:min(3,self.N)]))
        INFO("  Active gcs: {}.".format(self.active_gcs))
        INFO("  sd = %g, be = %g, ga = %g." % (self.sd, self.be, self.ga))
        INFO("  tau(mc) = %g, tau(gc) = %g, tau(pg) = %g." % (self.tau_mc, self.tau_gc, self.tau_pg))
        INFO("  leak(pg) = %g." % (self.leak_pg))
        INFO("  Noisy inputs = %g." % (self.noisy))

    def create_partitioned_sister_connectivity(self):
        if np.mod(self.N, self.S):
            raise ValueError("Number of granule cells {} must be divisible by number of sisters {} for partitioned connectivity.")
        INFO(f"Creating PARTITIONED connectivity for S={self.S} sisters.")                
        block_size = self.N//self.S
        self.W = np.zeros((self.M, self.N, self.S))        
        for s in range(self.S):
            inds = slice(block_size*s, block_size*(s+1))
            self.W[:,inds, s] = self.A[:,inds]
        self.connectivity = "partitioned"            
        return self.W
        
    def create_random_sister_connectivity(self):
        INFO(f"Creating RANDOM connectivity for S={self.S} sisters.")
        self.W = np.zeros((self.M, self.N, self.S))
        which_s = (np.random.rand(*self.A.shape)*self.S).astype(int) # HERE
        for s in range(self.S):
            self.W[:,:,s] = self.A*(s == which_s)
        self.connectivity = "random"
        return self.W

    def _setup_permuted_weight_matrices(self):
        self.Wisj  = np.array(np.transpose(self.W, (0,2,1)), order="C")
        self.Wis_j = np.reshape(self.Wisj, (-1, self.N),     order="C")
        self.Wjis  = np.array(np.transpose(self.W, (1,0,2)), order="C")
        self.Wj_is = np.reshape(self.Wjis, (self.N,-1),      order="C")

    def set_sister_cell_parameters(self, F=None, G=None, P=None, Q=None):
        self.sis_F  = F if F else np.ones((self.M, self.S), order = "C")
        self.sis_G  = G if G else np.ones((self.M, self.S), order = "C")
        self.sis_P  = P if P else np.ones((self.M, self.S), order = "C")
        self.sis_Q  = Q if Q else np.ones((self.M, self.S), order = "C")/self.S

        self.sis_PQ = np.sum(self.sis_P * self.sis_Q, axis=1)
    
    def _check_sister_parameters(self):
        if not hasattr(self, "S"):
            raise AttributeError("Number of sister cells is undefined. Run 'create_[partitioned|random]_sister_connectivity' first.")

        if self.S<=0:
            raise ValueError("Number of sister cells should be positive, is {}".format(self.S))

        params = [("sis_F", (self.M,self.S)), ("sis_G",(self.M,self.S)), ("sis_P",(self.M,self.S)), ("sis_Q",(self.M,self.S)), ("sis_PQ",(self.M,))]
        for p,p_shape in params:
            if not hasattr(self, p):
                raise AttributeError("Missing parameter {}. Run 'set_sister_cell_parameters'.".format(p))
            p_val = getattr(self, p)
            if type(p_val) != np.ndarray:
                raise TypeError("Expected type of {} to be numpy.ndarray, but is {}.".format(p, type(p_val)))
            if p_val.shape != p_shape:
                raise TypeError("Expected shape of {} to be {}, but it's {}.".format(p, p_shape, p_val.shape))

    def run_exact(self, y, solver=cvx.SCS, **kwargs):
        """ Uses CVXOPT to infer the odor by solving
        the convex optimization problem directly.
        """
        INFO("Running exact model.")
        with TimedBlock("RUNNING EXACT MODEL"):
            x = cvx.Variable(self.N)
            r = cvx.Variable(self.M)
            objective  = cvx.Minimize(cvx.norm(x,1)*self.be + 0.5*self.ga*cvx.sum_squares(x) + 0.5/self.sd/self.sd*cvx.sum_squares(r))
            constraint = [r == y - self.A @ x, x>=0]
            problem = cvx.Problem(objective, constraint)
            problem.solve(solver=solver, **kwargs)
            x_MAP = np.reshape(np.array(x.value),(self.N,))
            l_MAP = -constraint[0].dual_value
            r_MAP = l_MAP*self.sd*self.sd
        return {"x":x_MAP, "la": l_MAP, "r": r_MAP, "status":problem.status}
        
    def run_sister(self, odor, t_end, dt, keep_every = 1, Y_init = None, X_init = None, V_init = None, La_init = None, Mu_init = None, XFUN = linearly_rectify, report_frequency = 10, keep_till = np.inf):

        INFO(f"Run Parameters:")
        INFO(f"  Running to t={t_end}.")
        INFO(f"  Stepsize {dt=}.")
        INFO(f"  Keeping every {keep_every} step(s).")        
        t_all  = np.arange(0, t_end, dt)
        nt_all = len(t_all)
        INFO(f"  {nt_all} total time steps.")
        # We often want the sim to run for some large time interval T.
        # This is to e.g. get the final values of the inference.
        # But we only want it to periodically report within some shorter interval
        # in which we e.g. examine the dynamics.
        INFO(f"  Keeping times t < {keep_till}.")
        ind_keep = np.arange(0, min(keep_till, t_end), dt)[::keep_every]
        nt_keep  = len(ind_keep)
        INFO(f"  ({nt_keep} time steps to keep.)")

        self._check_sister_parameters()
        self._setup_permuted_weight_matrices()

        
        # Generate the input
        # - Y has every time step because we'll need this throughout the integration.
        # - 18 August 2020: Changed this to keeping only t_keep timesteps because
        #   the memory foot print is too large.
        
        with TimedBlock("PREPARING INPUTS"):
            Y  = np.zeros((nt_keep, self.M, self.S), order = "C")
            if Y_init is not None:
                Y[0, :,:] = Y_init

        with TimedBlock("SETTING INITIAL CONDITIONS"):
            V  = np.zeros((nt_keep, self.N), order="C")
            v  = V_init if V_init else np.zeros((self.N,))
            V[0,:] = v
                
            X  = np.zeros((nt_keep, self.N), order="C")
            x  = X_init if X_init else np.zeros((self.N,))
            X[0,:] = x
                
            La = np.zeros((nt_keep, self.M, self.S), order="C")
            la = La_init if La_init else np.zeros((self.M, self.S), order="C")
            la_ms = la.reshape((self.M*self.S,), order="C") # A vectorized view we'll need below.
            La[0, :, :] = la
            
            Mu = np.zeros((nt_keep, self.M, self.S), order="C")
            mu = Mu_init if Mu_init else np.zeros((self.M, self.S), order="C")
            Mu[0,:,:] = mu


        self.gc_mask = np.ones(self.N,) if self.active_gcs == "all" else np.array([i in self.active_gcs for i in range(self.N)])
        INFO(f"Set gc_mask according to active_gcs = {self.active_gcs}: {int(sum(self.gc_mask))}/{self.N} gcs active.")
        
        status    = "FAILED."
        success   = False
        last_iter = 0
        t_keep = [0]
        with TimedBlock("MAIN LOOP"):
            report_interval = nt_all//report_frequency
            t_report = 0
            i_keep   = 1
            for ti, t in enumerate(t_all):
                if ti == 0:
                    continue

                WX    = np.dot(self.Wisj, x) 
                WLa   = np.dot(self.Wj_is, la_ms) 

                if ti == 1:
                    Yt = Y[0]
                else:
                    y  = odor.value_at(t_all[ti-1], A = self.A) if hasattr(odor, "value_at") else odor
                    Yt = np.outer(y, np.ones(self.S))
                    if self.noisy:
                        Yt += np.random.randn(self.M, self.S) * self.noisy * self.sd * np.sqrt(dt)
                    
                dladt = -self.sis_F * la + (self.sis_G*Yt - self.S*WX - self.S * mu)/self.sd**2

                if self.leak_pg<np.inf:
                    dmudt = -self.leak_pg * mu + (la - self.sis_P*(np.sum(self.sis_Q * la, axis=1)/self.sis_PQ)[:,np.newaxis])

                dvdt  = -v + WLa
    
                la += dladt/self.tau_mc * dt

                if self.leak_pg<np.inf:
                    mu += dmudt/self.tau_pg * dt
                else:
                    mu = 0

                x  =  XFUN(v, self.be)/self.ga * self.gc_mask
                v +=  dvdt/self.tau_gc * dt    * self.gc_mask                 

                if np.any(np.isnan(la)):
                    status  = "Overflow at step {}.".format(t)
                    success = False
                    break
    
                last_iter = ti

                t_report += 1
                if t_report == report_interval:
                    n_prog = ti//report_interval
                    INFO("[" + "*" * n_prog + " "*(report_frequency - n_prog) + "]")
                    t_report = 0

                if (t < keep_till) and (not np.mod(ti, keep_every)):
                    X[ i_keep, :] = x
                    V[ i_keep, :] = v
                    La[i_keep, :] = la
                    Mu[i_keep, :] = mu
                    Y[ i_keep, :] = Yt
                    i_keep += 1
                    t_keep.append(t)

            
        if last_iter == nt_all-1: # All iterations ran
            status = "OK."
            success = True

        return {"T":np.array(t_keep), "X":X, "La":La, "Y":Y, "V":V, "Mu":Mu,"A10":self.A[0][:min(self.M,10)],
                "t_final":last_iter*dt,
                "x_final":x,
                "v_final":v,
                "y_final":y,
                "la_final":la,
                "mu_final":mu,
                "last_iter":last_iter, "success":success, "status":status}            


    def linearize(self, active_gcs):
        """
        Returns the matrix used in the linearization analysis
        for the case of 'active_gcs' being active.

        RETURNS
        A matrix H

        H: The linearization matrix in the base (la, mu, v) space.        
        
        H is arranged so that all the mcs (la) come first, 
        then the pgs (mu), then the gcs (v).
        
        """
        M,S,n = self.M, self.S, len(active_gcs)
        eta   = 1/self.ga/self.sd**2
        c_la  = 1/self.sd**2 

        al = self.A[:, active_gcs]
    
        dimH = (S + S )*M + n
        H    = np.zeros((dimH, dimH))
        # H is the dynamics matrix in the original (la, mu, v) space.
        # We will build H so that all the mitral cells go first
        # Then the Pg cells, then the granule cells.
        # The mitral cells and pg cells will have the sisters next to each other.    
        IND_MC = lambda m: slice(m*S, (m+1)*S)
        IND_PG = lambda m: slice(m*S + M*S, (m+1)*S + M*S)
        IND_GC = slice(2*M*S, 2*M*S+n)
        
        for m in range(M):
            Wm = self.W[m, active_gcs, :].T
            
            H[IND_MC(m), IND_MC(m)] = -np.eye(S)
            H[IND_MC(m), IND_PG(m)] = -S * c_la * np.eye(S)
            H[IND_MC(m), IND_GC   ] = -S * eta  * Wm
            H[IND_MC(m), :]        /= self.tau_mc

            H[IND_PG(m), IND_PG(m)] = -self.leak_pg * np.eye(S)
            H[IND_PG(m), IND_MC(m)] = np.eye(S) - np.ones((S,S))/S
            H[IND_PG(m), :]        /= self.tau_pg
    
            H[IND_GC   , IND_MC(m)] = Wm.T
    
        H[IND_GC, IND_GC] = -np.eye(n)
        H[IND_GC, :]     /= self.tau_gc
    
        self.IND_MC = IND_MC
        self.IND_PG = IND_PG
        self.IND_GC = IND_GC
        
        self.H    = H        
        
        return H

    def approximate_sparse_odour_eigs(self, nu):
        # nu is the Marchenko Pastur parameter
        INFO(f"Approximating sparse odour eigs using {nu=}.")
        eta = 1 / (self.ga * self.sd**2)
        kmu = self.tau_mc / self.tau_pg
        kv  = self.tau_mc / self.tau_gc
        # Eqn. roots0
        alpha2_pm = {s:(-(kv + self.ga * kmu) + s * csqrt((kv + self.ga * kmu)**2 - 4 * nu * self.ga * kv * kmu / self.S))/2 for s in [-1, 1]}
        # Eqn. alpha_1pm
        alpha1_pm = {s:-1/(2*np.sqrt(eta * self.S)) * \
                     ((2 + kv + self.leak_pg * kmu)*alpha2_pm[s] + kv + self.ga*kmu + (self.ga + self.leak_pg) * kmu * kv) / \
                     (s * np.sqrt((kv + self.ga*kmu)**2 - 4 * nu * self.ga * kv * kmu / self.S)) \
                     for s in [-1,1]}
    
        return {"high":np.array([np.sqrt(eta * self.S)/self.tau_mc * (alpha1_pm[-1] + s * csqrt(alpha2_pm[-1])) for s in [-1, 1]]),
                "low":np.array([np.sqrt(eta * self.S)/self.tau_mc * (alpha1_pm[+1] + s * csqrt(alpha2_pm[+1])) for s in [-1, 1]])}


    def approximate_dense_odour_eigs(self, nu):

        approx_nu = self.approximate_sparse_odour_eigs(nu)
        qxi_low_roots_approx, qxi_high_roots_approx = [approx_nu[fld] for fld in ["low", "high"]]

        # Now compute the approximations for q0xi
        approx_0  = self.approximate_sparse_odour_eigs(0)
        q0xi_low_roots_approx, q0xi_high_roots_approx = [approx_0[fld] for fld in ["low", "high"]]        

        # There's only one root in this case, as alpha^2_+ 
        # in Eqn high_low = 0, so its plus and minus square
        # root values are zero, and we take one of them
        # (but as a singleton array, hence [:1])
        q0xi_low_root_approx = q0xi_low_roots_approx[:1] 

        return {"qxi_high_roots_approx":qxi_high_roots_approx,
                "qxi_low_roots_approx":qxi_low_roots_approx,
                "q0xi_high_roots_approx":q0xi_high_roots_approx,
                "q0xi_low_root_approx":q0xi_low_root_approx}

            
    
def get_x_MAP(run_params_json):
    with open(run_params_json, "rb") as in_file:
        p = json.load(in_file)

    ob = OlfactoryBulb(**p)
    y = np.dot(ob.A, get_x_true(p["N"],p["k"], spread=p["spread"]))
    res = ob.run_exact(y, eps=5e-13)
    return res["x"]

    
