# This script contains the fucntions for loading the data for and plotting the effects of odour denseity
import logging
import numpy as np
from numpy import *

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib import cycler
from matplotlib import ticker
import inspect

import util
from olfactory_bulb import get_x_true
from olfactory_bulb import OlfactoryBulb
import datatools as dt
import figtools as ft

logger = util.create_logger("effect_of_density")
INFO   = logger.info

plt.style.use("default")
plt.style.use("./ms.mplstyle")

rand = np.random.rand

def load_data(t_on = 0.1, S = 8, n_max = 20, data = None, t_decay = [2.6, 2.8], k_vals = list(range(1,11))):

    INFO("Started {}.".format(inspect.stack()[0][3]))

    with dt.SilenceLoggers("olfactory_bulb"):
        if not data:
            data = {k: dt.load_Sk(S, k, root_folder = "sweep_S_k_long", n_max = n_max) for k in k_vals}
    
        # The elements of data are (trials_data, t, params_files) tuples
        
        t = data[k_vals[0]].t
        trials_data = {k:d.results_array for k,d in data.items()}
    
        # Data is a dictionary indexed by k.
        # Each element contains n_max trials
        # Each trial then has the X,La,T etc. fields    
        INFO("{} time indices from {:1.3f} - {:1.3f} seconds.".format(len(t), t[0], t[-1]))
        
        fs = int(round(1/(t[1] - t[0])))
        INFO(f"fs: {fs}")    
    
        n_true   = array(k_vals)    
        n_hat    = array([[sum(td["x_MAP"]>1e-2) for td in trials_data[n]] for n in sorted(n_true)])
    
        INFO("Computing decay time constants.")
        decay_slc = slice(int(t_decay[0]*fs), int(t_decay[1]*fs))
        INFO("Using time window {:1.3f} - {:1.3f} sec. (samples {} - {}).".format(t_decay[0], t_decay[1], decay_slc.start, decay_slc.stop))    
        decay_sub = lambda X: X[decay_slc]
        n_decay   = decay_slc.stop - decay_slc.start
        tdecay    = arange(n_decay)/fs
    
        # Each element of D is time x trials array showing the rmse timecourse
        D      = {n:array([r["x_rmse"] for r in trials_data[n]]).T for n in n_true}
        # Each element of decays is a vector of time constants, one for each trial
        decays = {n:-1/polyfit(tdecay, log(decay_sub(Dn)), 1)[0] for n, Dn in D.items()}

    INFO("Finished {}.".format(inspect.stack()[0][3]))    
    return {"t":t, "trials_data":trials_data, "D":D, "decays":decays, "n_hat":n_hat, "n_true":n_true, "data":data}


def plot_effect_of_density(results, ton = 0.1, n_low = 3, n_high = 10, xlim_decay_high = [0.4, 0.6], xlim_decay_low = [0.4, 0.6], ylim_decay=None, which_trial_low=0, which_trial_high=0):
    INFO("Started {}.".format(inspect.stack()[0][3]))
    t = results["t"]
    trials_data = results["trials_data"]

    fig = plt.figure(figsize=(8,4))
    ft.apply_styles()
    
    gs = GridSpec(2,6)
    ax = []

    n_vals = list(trials_data.keys())
    cmap = ft.cm_sub(cm.GnBu, 0.2, 0.8)

    n_cols = {n:cmap(n/n_vals[-1]) for n in n_vals}
    n_cols_list = array([n_cols[n] for n in n_vals])

    n_true  = results["n_true"]
    n_hat   = results["n_hat"]
    n_hat_m = mean(n_hat,axis=1)
    n_hat_s = std(n_hat, axis=1)

    D      = results["D"]
    decays = results["decays"]

    lgnd_frmt = {"fontsize":8, "frameon":False, "labelspacing":0, "borderpad":0, "borderaxespad":0.5}
    
    INFO("Plotting high density MAP solution.")    
    ax.append(plt.subplot(gs[0, :3]))
    plt.plot(get_x_true(1200,n_high),":",color="lightgray", linewidth=1)
    plt.plot(trials_data[n_high][which_trial_high]["x_MAP"], color = n_cols[n_high])
    plt.xlim(0,1200)
    plt.ylim(-0.01,1.01)
    xt = [i*200 for i in range(7)]
    ax[-1].set_xticks(xt)
    plt.xlabel("Molecule index")
    plt.ylabel("Concentration")
            
    INFO("Plotting recovered vs. true density.")
    ax.append(plt.subplot(gs[1, :3]))
    pfun = "loglog"
    ft.plot0(n_true, n_true, col_cyc="lightgray", ax=ax[-1], plot_fun = pfun, plot_args={"linestyle":":", "label":"$n_{true} = n_{rec.}$"})
    ft.plot0(array([n_true, n_true]), array([n_hat_m-n_hat_s, n_hat_m+n_hat_s]),
             col_cyc = cycler(color=n_cols_list), ax=ax[-1], plot_fun = pfun, plot_args={"linewidth":2})
    ft.plot0(n_true, n_hat_m, err=n_hat_s,
             col_cyc = cycler(color=n_cols_list), mode="scatter",
             ax=ax[-1], plot_fun = pfun,
             plot_args={"markersize":4, "marker":"o"}, err_bar_args={"linewidth":2},
             ylabel="$n$ (recovered)")
    plt.xlabel("$n$ (true)", labelpad=-1)
    plt.legend(**lgnd_frmt)
    
    INFO("Plotting time course of recovery.")
    n_plot = [3,6,10]
    h = []
    ax.append(plt.subplot(gs[:, 3:]))            
    for i, nn in enumerate(n_plot):
        ni = list(n_true).index(nn)
        h.append(ft.plot0(t - ton, D[ni]/D[ni][0],
                 ax = ax[-1], col_cyc = ft.set_alpha(0.5)(n_cols[ni]),
                 plot_fun = "semilogy",
                 xlim=(-0.1,1.4), xticks=[0,0.5,1,1.5],
                 ylim=(1e-10,10),
                 xlabel="Time / sec.", ylabel="RMS Error",
                          plot_args={"linewidth":1})[1][0])
    plt.legend(h[-len(n_plot):], ["$n$ = " + str(nn) for nn in n_plot],  **lgnd_frmt)

    # INFO("Plotting decay rates vs density.")    
    # ax.append(plt.subplot(gs[1, 3:]))
    # iperm     = np.random.permutation(prod(n_hat.shape))
    # cols_base = [ft.set_alpha(0.75)(n_cols_list[i//n_hat.shape[1]]) for i in range(prod(n_hat.shape))]
    # cols_perm = [cols_base[i] for i in iperm]

    # all_n_hat  = n_hat.flatten()
    # all_decays = array([d for n,d in decays.items()]).flatten()
    
    # ft.plot0(all_n_hat[iperm], all_decays[iperm] * 1000,
    #          ax = ax[-1], col_cyc=cycler(color=cols_perm),
    #          mode="scatter",
    #          plot_args={"markersize":5, "marker":"o", "markeredgewidth":0},
    #          xlabel="$n$ (recovered)", ylabel = "$\\tau$ (ms)")
    # ax[-1].set_xticks([i*10 for i in range(0,7)])

    # INFO("Plotting estimates of decay time constants for the sparse and dense settings.")
    # p = dt.load_default_params()
    # ob = OlfactoryBulb(**p)

    # eigs_sparse = ob.approximate_sparse_odour_eigs(nu=1)
    # eigs_dense  = ob.approximate_sparse_odour_eigs(nu=55/50)
    # # -1/Real part of Eqn. qxi_low_roots_approx
    # # 2 * (p["ga"]*p["tau_mc"]*p["tau_gc"] + p["tau_mc"] * p["tau_pg"])/(p["ga"] * (p["tau_mc"] + p["tau_gc"]) + p["tau_pg"])    
    # tau_sparse = -1/np.real(eigs_sparse["low"][0]) 
    # # tau_dense  = p["tau_gc"] + p["tau_pg"]/p["ga"]    

    # xl = plt.xlim()
    # plt.plot(xl, [tau_sparse * 1000]*2, ":",   linewidth=1.5, label="$\\tau$ (pred.)",  color="lightgray")
    # # plt.plot(xl, [tau_dense]*2,  "--",  linewidth=1.5, label="$\\tau$ (dense)",   color="lightgray")    
    # plt.xlim(xl)
    # plt.ylim([50,80])
    # plt.legend(**lgnd_frmt)

    INFO("Finalizing figure.")
    plt.tight_layout()
    ft.label_axes(ax, "ABCD",fontsize=14, verticalalignment="center", horizontalalignment="left",fontweight="bold")    
    fig_file = "effect_of_density.pdf"
    INFO(f"Saving as {fig_file}.")    
    plt.savefig(fig_file, bbox_inches="tight")
    
    INFO("Finished {}.".format(inspect.stack()[0][3]))        
    
