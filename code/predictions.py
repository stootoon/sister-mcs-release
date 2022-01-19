import os
import logging
import numpy as np
from numpy import *
from scipy.interpolate import interp1d
from scipy.stats import spearmanr

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
import inspect

import pickle
import pdb
import datatools as dt
import figtools as ft
import util

logger = util.create_logger("predictions")
INFO   = logger.info

plt.style.use("default")
plt.style.use("./ms.mplstyle")

def gen_monotonic_function_gp(x_rng=[-1,1], nx=101, sd=1, num_funs = 10, margin=1.05, seed = 0,
                              zero_mean = True):
    random.seed(seed)
    x  = linspace(x_rng[0], x_rng[1], nx)
    dx = mean(diff(x))
    ij = np.subtract.outer(x,x);
    K  = exp(-ij**2/sd**2/2)
    U,S,_ = linalg.svd(K)
    L  = U @ diag(sqrt(S));
    X  = random.randn(L.shape[0],num_funs)
    Y  = L @ X
    dY = diff(Y,axis=0).min(axis=0)/dx
    Y -= margin*outer(x,np.minimum(dY,0))
    Y -= Y.min(axis=0)
    Y /= Y.max(axis=0)
    Y *= diff(x_rng)
    if zero_mean:
        Y -= Y.mean(axis=0)

    return [interp1d(x, yi) for yi in Y.T], x

def load_data(which_seed = 0, which_eps = [0,1,2], mon_x_rng=[-10,10], mon_nx=101, mon_sd=1, mon_margin=1.05, mon_seed=0, zero_mean = False):
    INFO("Started {}.".format(inspect.stack()[0][3]))
    root_folder = dt.data_folder("sweep_more_random_odours")        
    sweep = dt.FreeSweepDataset(root_folder, params = ["leak_pg", "seed"], load_on_demand = False)
    odours_data = sweep.load_sweep(warn_on_missing = False, vars = ["la_final"])
    INFO("Loading la_final.")
    la_all = {k:array([odi["la_final"] for odi in od]) for k,od in odours_data.items()}
    n_odours, M, S = la_all[(which_eps[0], which_seed)].shape
    INFO(f"la_final data found with {M=}, {S=}, {n_odours=}.")

    INFO(f"Generating monotonic functions using x_rng={mon_x_rng}, nx={mon_nx}, mon_sd={mon_sd}, margin={mon_margin}, seed={mon_seed}.")
    funs, mon_x = gen_monotonic_function_gp(x_rng = mon_x_rng, nx = mon_nx, sd = mon_sd, margin = mon_margin, seed = mon_seed, num_funs = M * S, zero_mean = zero_mean)           
    mons_sg = [[funs[s*M+g] for g in range(M)] for s in range(S)]

    INFO(f"Applying nonlinearities to model mitral cells to get experimentally recorded values.")    
    # 'transpose' below is to put in odour,glom,sis order like la_all
    la_all_mon ={k:array([[mons_sg[s][g](la[:,g,s]) for g in range(M)] for s in range(S)]).transpose([2,1,0]) for k,la in la_all.items()}

    INFO(f"Computing spearman correlations for {which_seed=}.")
    sr = {}
    for ee in which_eps:
        INFO(f"ε = {ee}")
        la_ogs = array(la_all_mon[(ee,which_seed)])
        sr[ee] = array([[spearmanr(la_ogs[:,g,i], la_ogs[:,g,j])[0] for g in range(M)] for i in range(S-1) for j in range(i+1,S)])
    la_ogs = array(la_all_mon[(0,which_seed)])

    INFO(f"Computing spearman correlations for non-sisters and ε = 0.")    
    sr["diff_glom"] = array([[spearmanr(la_ogs[:,g,i], la_ogs[:,g+1,j])[0] for g in range(M-1)] for i in range(S-1) for j in range(i+1,S)])
    
    INFO(f"Computing sister indices.")
    ij = [[i,j] for i in range(S-1) for j in range(i+1,S)]
    
    INFO(f"Computing spearman correlation distribution stats.")
    stats = {k:array(list(percentile(x.flatten(),[5,50,95]))+[np.min(x)]) for k,x in sr.items()}
    
    INFO("Done {}.".format(inspect.stack()[0][3]))            
    return {"sr":sr,
            "ij":ij,
            "la_eps":{e:la_all[(e, which_seed)] for e in which_eps},
            "la_eps_mon":{e:la_all_mon[(e, which_seed)] for e in which_eps},
            "mons_sg":mons_sg,
            "stats":stats}

def get_data_and_example(data, which_example = "min"):
    INFO("Started {}.".format(inspect.stack()[0][3]))    

    assert which_example in ["med","min"], f"{which_example=} must be one of 'med','min'."

    INFO("Loading correlations data.")
    sr, ij, la_eps, la_eps_mon, stats, mons_sg = [data[k] for k in ["sr", "ij", "la_eps", "la_eps_mon", "stats", "mons_sg"]]
    INFO(f"Found data for ε = {list(la_eps.keys())}.")
    
    if which_example == "min":
        row_min, glom_min = [int(w) for w in where(sr[2] == sr[2].min())]
    elif which_example == "med":
        sr_med = np.median(sr[2])
        d_med  = np.abs(sr[2] - sr_med)
        rows, gloms = where(d_med == d_med.min()) # The median is achieved in more than one place
        INFO(f"Found {len(rows)} comparisons achieving the median.")
        ind_use = 7 #6 #3
        INFO(f"Using #{ind_use}.")
        row_min, glom_min = [int(w) for w in (rows[ind_use], gloms[ind_use])]
        
    sis_i,sis_j       = ij[row_min]
    INFO(f"{'Minimum' if which_example=='min' else 'Median'} correlation of {sr[2].min() if which_example == 'min' else sr_med:.4f} found for sisters {sis_i=} and {sis_j=} of {glom_min=}.")

    la_ogs            = la_eps[2] # Get the value for ε = 2
    la_ogs_mon        = la_eps_mon[2]
    n_odours, G, S    = la_ogs.shape
    r                 = spearmanr(la_ogs_mon[:,glom_min,sis_i], la_ogs_mon[:,glom_min,sis_j])[0]

    if which_example == "min":
        assert np.abs(r-stats[2][-1])<1e-12, "Correlations don't match."
    else:
        INFO(f" Median correlation: {sr_med:1.6f}.")
        INFO(f"Example correlation: {r:1.6f}.")

    INFO("Finished {}.".format(inspect.stack()[0][3]))        
    return la_ogs, la_ogs_mon, glom_min, sis_i, sis_j, stats, mons_sg

def plot_predictions(data, order = [0,1,2,"diff_glom"], figsize=(6,5), which_example = "min", ylabel_args = {}, plabel_args = {}):
    INFO("Started {}.".format(inspect.stack()[0][3]))

    la_ogs, la_ogs_mon, glom_min, sis_i, sis_j, stats, mons_sg = get_data_and_example(data, which_example)
    n_odours, G, S    = la_ogs.shape    
    INFO(f"{n_odours=}, {G=}, {S=}.")

    plt.figure(figsize=figsize)
    ft.apply_styles()
    n_rows, n_cols = 8, 2

    gs = GridSpec(n_rows, n_cols)

    n_inset_rows = 2    
    ax_full   = plt.subplot(gs[n_inset_rows:,0])
    ax_inset  = plt.subplot(gs[:n_inset_rows,0])
    for ax_name, ax in zip(["full","inset"], [ax_full, ax_inset]):
        INFO(f"Plotting spearman correlation, {ax_name}.")    
        h  = ax.bar(arange(4), [stats[k][1] for k in order],width=0.7)
        which_cm = cm.Greens
        cols = [which_cm(1.), which_cm(0.8), which_cm(0.6), cm.gray(.5)]
        [hi.set_facecolor(col) for hi, col in zip(h,cols)]
        ax.plot(outer([1,1],arange(4)), [[stats[k][m] for k in order] for m in [0,2]], "r", lw=1)
        ax.axhline(0,linestyle=":",color="gray")

        ax.set_ylim([min(ax.get_ylim()),1.02] if ax_name == "full" else [0.98,1.002])
        ax.set_xticks(arange(len(order)))
        ax.set_xticklabels(labels=[("$\\varepsilon$" + f" = {v}") if type(v) is not str else ("$\\varepsilon$ = 0" + "\n(non-sisters)") for v in order])
        ax.tick_params(axis='x',which='major',pad=0)
        ax_name == "full" and ax.set_ylabel("Spearman's rank correlation $\\rho$")
        ax_name == "inset" and ax.set_ylabel("$\\rho$")
        ax_name == "inset" and ax.plot(arange(4)[:-1], [stats[k][-1] for k in order][:-1], "r.")

    plt.gcf().align_labels([ax_full, ax_inset])

    INFO(f"Plotting responses of sister {sis_i}_0 vs {sis_j}_0 (Note: labels are 1-based)")
    ax_sis = []
    for i, la in enumerate([la_ogs, la_ogs_mon]):
        ax_sis.append(plt.subplot(gs[n_rows//2:, 1] if i else gs[:n_rows//2,1]))        

        la_so  = la[:,glom_min,:]
        lai    = la_so[:, sis_i]
        laj    = la_so[:, sis_j]

        which_odour = argmax(abs(la_so[:, sis_i] - la_so[:,sis_j]))
        ax_sis[-1].plot(lai, laj, "o", markersize=2, color=cols[2],lw=1)
        #ax_sis[-1].plot(lai[which_odour], laj[which_odour], "o", markersize=4, color="r")
        #xl = ax_sis[-1].get_xlim()
        #yl = ax_sis[-1].get_ylim()
        i==0 and ax_sis[-1].axhline(0, linestyle=":", color="lightgray", linewidth=1, zorder=-1)
        i==0 and ax_sis[-1].axvline(0, linestyle=":", color="lightgray", linewidth=1, zorder=-1)    
        ax_sis[-1].set_xlabel(f"Sister cell {sis_i+1} response",labelpad=0) 
        ax_sis[-1].set_ylabel(f"Sister cell {sis_j+1} response",labelpad=1)
        ax_sis[-1].axis("square")
        ax_sis[-1].set_xlim([0,12] if i else [-4,4])
        ax_sis[-1].set_ylim([0,12] if i else [-4,4])
        ax_sis[-1].set_xticks(arange(0,13,4) if i else arange(-4,5,2))
        ax_sis[-1].set_yticks(arange(0,13,4) if i else arange(-4,5,2))

    plt.gcf().align_labels(axs=ax_sis)        
    
    INFO(f"Plotting sister cell nonlinearities.")            
    ax3 = ax_sis[-1].inset_axes([2.5,8.25,3.5,3.5], transform = ax_sis[-1].transData)
    t = linspace(-6, 6, 101);
    sis_cols = [cm.Blues(0.5), cm.Blues(0.8)]
    mon_i = mons_sg[sis_i][glom_min]
    mon_j = mons_sg[sis_j][glom_min]
    yi = mon_i(t)
    yj = mon_j(t)    
    ax3.plot(t, yi, color=sis_cols[0], label=f"Sis. {sis_i+1}", linewidth=1)
    ax3.plot(t, yj, color=sis_cols[1], label=f"Sis. {sis_j+1}", linewidth=1)

    ax3.set_xticks(arange(-5,5.1,5))
    ax3.set_yticks(arange(0,12.1,5))
    leg = ax3.legend(loc='upper left', frameon=False, labelspacing=0, fontsize=6,
                     borderpad=0, handlelength=1, borderaxespad=0.3)

    ax3.set_xlabel("Model ($\lambda_i^s$)", fontsize=6, labelpad=-1);
    ax3.set_ylabel("Experiment ($\\tilde \lambda_i^s$)",fontsize=6,labelpad=-2)
    ax3.tick_params(axis='both', labelsize=6)

    plt.tight_layout(h_pad=1.5, w_pad=0.5)
    ft.label_axes([ax_inset]+ax_sis,"ABCDE",fontsize=14,
                  verticalalignment="center", horizontalalignment="left",
                  fontweight="bold", **plabel_args)

    output_file = "spearman_rank_new.pdf"    
    
    plt.savefig(output_file,bbox_inches="tight")
    INFO(f"Wrote {output_file}.")
    INFO("Finished {}.".format(inspect.stack()[0][3]))


def plot_responses(data, figsize=(6,4.5), which_odours = [0,1], which_example = "min"):
    INFO("Started {}.".format(inspect.stack()[0][3]))

    la_ogs, la_ogs_mon, glom_min, sis_i, sis_j, stats, mons_sg = get_data_and_example(data, which_example)
    n_odours, G, S    = la_ogs.shape    
    INFO(f"{n_odours=}, {G=}, {S=}.")
    
    plt.figure(figsize=figsize)
    ft.apply_styles()
    n_rows, n_cols = 2, 2

    gs = GridSpec(n_rows, n_cols)
    sis_cols = [cm.Blues(0.5), cm.Blues(0.8)]

    ax = []
    for i, la in enumerate([la_ogs, la_ogs_mon]):
        la_so  = la[:,glom_min,:]
        for j, which_odour in enumerate(which_odours):
            INFO(f"Plotting sister cell activities for odour {which_odour} using {'transformed' if i else 'original'} responses .")

            if j:
                ax.append(plt.subplot(gs[i,j], sharey=ax[-1]))
            else:
                ax.append(plt.subplot(gs[i,j]))
            cols = [cm.gray(0.6) if i not in [sis_i, sis_j] else (sis_cols[0] if i == sis_i else sis_cols[1]) for i in range(S)]
            lws  = [1 if i not in [sis_i, sis_j] else 2 for i in range(S)]
            ax[-1].set_prop_cycle(color= cols, linewidth=lws)
            ax[-1].plot(outer([1,1],arange(1,S+1)), [0*la_so[which_odour], la_so[which_odour].T])
            ax[-1].axhline(0, linestyle=":", color="gray", lw=1, zorder=-1)
            ax[-1].set_xticks([1]+list(range(5,26,5)))
            (i >= 0) and ax[-1].set_xlabel("Sister cell")
            #(j == 0) and ax[-1].set_ylabel("Activity\nat convergence",labelpad=0)
            yt = ax[-1].get_yticks()
            (i == 0) and ax[-1].set_title(f"Odour {j+1}")            
            (j == 0) and (i==0) and ax[-1].set_ylabel("Model ($\lambda_i^s$)", labelpad=0)
            (j == 0) and (i==1) and ax[-1].set_ylabel("Experiment ($\\tilde \lambda_i^s$)", labelpad=0)
            
            #ax[-1].set_yticklabels(["" if mod(yti,1) else f"{yti:g}" for yti in yt])
            #ax[-1].set_yticklabels([f"{yt:g}" if yt.endswith(".0") else "" for yt in [yti.get_label() for yti in ax[-1].get_yticklabels()]])
    
    plt.tight_layout(h_pad=1, w_pad=1)
    ft.label_axes(ax,"ABCD",fontsize=14,
                  verticalalignment="center", horizontalalignment="left",
                  fontweight="bold", dy=0.02,dx=-0.02)

    output_file = "monotonic_responses.pdf"
    plt.savefig(output_file,bbox_inches="tight")
    INFO(f"Wrote {output_file}.")
    INFO("Finished {}.".format(inspect.stack()[0][3]))
    
    
    


