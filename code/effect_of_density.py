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

import pdb

logger = util.create_logger("effect_of_density")
INFO   = logger.info

plt.style.use("default")
plt.style.use("./ms.mplstyle")

rand = np.random.rand

def load_data(t_on = 0.1, S = 8, n_max = 20, data = None, t_decay = [2.6, 2.8], k_vals = list(range(1,11)), M_vals=[50]):

    def load_results_for_M(M):
        if M == 50:
            data = {k: dt.load_Sk(S, k, which_M = M, vars_to_load=["x_MAP","x_final", "X"], drop_vars = ["X"], root_folder = "sweep_S_k_long", n_max = n_max) for k in k_vals}
        else:
            data = {k: dt.load_Sk(S, k, which_M = M, vars_to_load=["x_MAP","x_final", "X"], drop_vars = ["X"], root_folder = "sweep_S8_M_k_long", n_max = n_max) for k in k_vals}
    
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

        keep_till = trials_data[n_true[0]][0]["params"]["keep_till"]
        if t_decay[1] > keep_till:
            INFO(f"Not computing decays because data kept till {keep_till:.3f} < end of desired decay interval, {t_decay[1]:.3f}.")
            D = None
            decays = None
        else:
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

        return {"t":t, "trials_data":trials_data, "D":D, "decays":decays, "n_hat":n_hat, "n_true":n_true, "data":data}
    
    INFO("Started {}.".format(inspect.stack()[0][3]))

    with dt.SilenceLoggers("olfactory_bulb"):
        results = {M:load_results_for_M(M) for M in M_vals}

    INFO("Finished {}.".format(inspect.stack()[0][3]))    
    return results


def plot_effect_of_density(data, ton = 0.1, n_low = 3, n_high = 10, M_vals = [50], xlim_decay_high = [0.4, 0.6], xlim_decay_low = [0.4, 0.6], ylim_decay=None, which_trial_low=0, which_trial_high=0):
    INFO("Started {}.".format(inspect.stack()[0][3]))
    base_M = M_vals[0]
    results = data[base_M]
    
    t = results["t"]    
    trials_data = results["trials_data"]

    fig = plt.figure(figsize=(8,4))
    ft.apply_styles()
    
    gs = GridSpec(2,6)
    ax = []

    def cols_list_fun(cmap, n_vals, index = False):
        inds = n_vals if not index else range(len(n_vals))
        n_cols      = {n:cmap((i - inds[0])/(inds[-1] - inds[0])) for i,n in zip(inds,n_vals)}
        n_cols_list = array([n_cols[n] for n in n_vals])
        return n_cols, n_cols_list

    n_vals = list(trials_data.keys())
    cmap = ft.cm_sub(cm.GnBu, 0.2, 0.8)

    n_cols, n_cols_list = cols_list_fun(cmap, n_vals)
    
    M_cmap = ft.cm_sub(cm.Greys_r, 0, 0.7)
    M_cols, M_cols_list = cols_list_fun(M_cmap, M_vals, index= True)
    M_cols[base_M] = n_cols_list[-1]
    M_cols_list[0] = n_cols_list[-1]    

    n_true  = {M:data[M]["n_true"] for M in M_vals}
    n_hat   = {M:data[M]["n_hat"]  for M in M_vals}
    n_hat_m = {M:mean(n_hat[M],axis=1) for M in M_vals}
    n_hat_s = {M:std(n_hat[M], axis=1) for M in M_vals}

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
    pfun = "plot"

    hM = []
    ft.plot0(array([n_true[base_M], n_true[base_M]]), array([n_hat_m[base_M]-n_hat_s[base_M], n_hat_m[base_M]+n_hat_s[base_M]])/n_true[base_M],
             col_cyc = cycler(color=[M_cols[base_M]]),
             ax=ax[-1], plot_fun = pfun, plot_args={"linewidth":2})
    _, h = ft.plot0(n_true[base_M], n_hat_m[base_M]/n_true[base_M], err=n_hat_s[base_M]/n_true[base_M],
             col_cyc = cycler(color=[M_cols[base_M]]), mode="plot",
             ax=ax[-1], plot_fun = pfun,
             plot_args={"markersize":4, "marker":"o","label":f"M = {base_M}"}, err_bar_args={"linewidth":2},
             ylabel="$n$ (rec.) / $n$")

    hM.append(h)
    for iM, M in enumerate(sorted(M_vals)):
        if M != base_M:
            # ft.plot0(array([n_true[M], n_true[M]]), array([n_hat_m[M]-n_hat_s[M], n_hat_m[M]+n_hat_s[M]])/n_true[M],
            #          col_cyc = cycler(color=[M_cols[M]]),
            #          ax=ax[-1], plot_fun = pfun, plot_args={"linewidth":2})
            _, h = ft.plot0(n_true[M], n_hat_m[M]/n_true[M], err=n_hat_s[M]/n_true[M],
                     col_cyc = cycler(color=[ft.set_alpha(0.75)(M_cols[M])]), mode="plot",
                     ax=ax[-1], plot_fun = pfun,
                     plot_args={"label":f"M = {M}", "markersize":2, "marker":"o","linewidth":1.5,"zorder":-iM}, err_bar_args={"linewidth":1},
                     ylabel="$n$ (recovered) / $n$ ")
            hM.append(h)
    plt.yticks([1,5,9])
    plt.xlabel("$n$", labelpad=-1)
    plt.legend(**lgnd_frmt)
    
    INFO("Plotting time course of recovery.")
    n_plot = [3,6,10]
    h = []
    ax.append(plt.subplot(gs[:, 3:]))            
    for i, nn in enumerate(n_plot):
        ni = list(n_true[base_M]).index(nn)
        h.append(ft.plot0(t - ton, D[ni]/D[ni][0],
                 ax = ax[-1], col_cyc = ft.set_alpha(0.5)(n_cols[ni]),
                 plot_fun = "semilogy",
                 xlim=(-0.1,1.4), xticks=[0,0.5,1,1.5],
                 ylim=(1e-10,10),
                 xlabel="Time (sec.)", ylabel="RMS error",
                          plot_args={"linewidth":1})[1][0])
    plt.legend(h[-len(n_plot):], ["$n$ = " + str(nn) for nn in n_plot],  **lgnd_frmt)

    INFO("Finalizing figure.")
    plt.tight_layout()
    ft.label_axes(ax, "ABCD",fontsize=14, verticalalignment="center", horizontalalignment="left",fontweight="bold")    
    fig_file = "effect_of_density.pdf"
    INFO(f"Saving as {fig_file}.")    
    plt.savefig(fig_file, bbox_inches="tight")
    
    INFO("Finished {}.".format(inspect.stack()[0][3]))        
    
