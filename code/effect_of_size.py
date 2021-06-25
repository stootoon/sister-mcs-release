# This script contains the fucntions for loading the data for and plotting the effects of system size (M,N)
import os, sys
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
import datatools as dt
import figtools as ft

import pdb

logger = util.create_logger("effect_of_size")
INFO   = logger.info

plt.style.use("default")
plt.style.use("./ms.mplstyle")

rand = np.random.rand


def load_data():
    # S = 25
    INFO("Started {}.".format(inspect.stack()[0][3]))

    root_folder = dt.data_folder("sweep_MNk_conc_spread")    
    sweep = dt.FreeSweepDataset(root_folder, params = ["M","N","k"], load_on_demand = False)
    sweep.load_sweep([(100, 1200, 3), (100, 4800, 3), (200, 1200, 3), (200, 4800, 3)])

    INFO("Finished {}.".format(inspect.stack()[0][3]))    
    return sweep


def plot_effect_of_size(figure_data, t_on = 0.1, cm_mc = ft.pop_cmaps["mc"], cm_gc=ft.pop_cmaps["gc"], glom_target = 0.5, which_trial = 0, xl=(-0.05,0.4)):
    INFO("Started {}.".format(inspect.stack()[0][3]))
    sweep = figure_data
    plt.figure(figsize=(8,6)); gs = GridSpec(3,4)
    all_ax = {}
    # Pick glomeruli whose final value is near 1.
    # This is to make it easier to plot their behaviour on the same axes
    which_glom = {k:argmin(abs(swp[which_trial]["La"][-1,:,0]-glom_target)) for k, swp in sweep._data.items()}
    INFO(f"Glomeruli whose final activity is closest to {glom_target}: {which_glom}.")
    mc_col = cm_mc(0.5)
    mc_col_trace = cm_mc(0.01)
    gc_col = cm_gc(0.5)
    gc_col_trace = cm_gc(0.01)    
    xt   = arange(0,xl[-1]+0.01,0.1)
    for icol, (M,N,k) in enumerate(sweep.get_param_value_combs_loaded()):
        data = sweep[M,N,k][which_trial]
        tt   = data["T"]
        x_MAP = data["x_MAP"]
        for irow, content in enumerate(["MC","GC","RMSE"]):
            ax = plt.subplot(gs[irow, icol])
            all_ax[irow, icol] = ax
            if content == "MC":
                yl = [-2.1,2.1]
                yt = [-2,-1,0,1,2]
                cols = [mc_col_trace]*25
                cols[-1] = mc_col
                col_cyc = cycler(color=cols)
                La = data["La"][:, which_glom[M,N,k],:]
                ft.plot0(tt - t_on, 0*tt, ax=ax, col_cyc="lightgray", plot_args={"linewidth":1})
                _, h = ft.plot0(tt - t_on, La,
                                ax = ax, col_cyc = col_cyc, yticks = yt,
                                ylim = yl, xlim = xl, xticks=xt, plot_args={"linewidth":1})
                ax.set_yticklabels([f"{y:g}" for y in yt])
                ax.set_xticklabels([f"{lab:g}" for lab in ax.get_xticks()])
                (icol == 0) and plt.ylabel("MC activity", labelpad=0)
                (irow == 0) and plt.title(f"{M=}, {N=}".format(M,N,k), fontsize=9)
                plt.xlabel("Time / sec.")
            elif content == "GC":
                n_out = 10
                cols = []
                for ik in range(k):
                    cols.append(gc_col) # gc_col_trace if ik < k-1 else gc_col)
                for i in range(n_out):
                    cols.append(ft.set_alpha(0.5)(cm.Greys(rand()*0.8+0.2)))
    
                col_cyc = cycler(color=cols)
                i_in = np.where(get_x_true(N,k))[0]
                i_out = argsort(-np.max(data["X"],axis=0))[k:k+n_out]
                i_both = list(i_in) + list(i_out)

                yl = [-0.1,3.05]
                yt = arange(0,3.1,1)                
                _, h = ft.plot0(tt - t_on, data["X"][:,i_both], ax = ax,
                                col_cyc = col_cyc, ylim = yl,
                                yticks=yt, xlim = xl, xticks=xt,
                                plot_args={"linewidth":1})
                
                [plt.plot(plt.xlim()[1], x, "r<", linewidth=1) for x in x_MAP[x_MAP>0.5]]
                (icol == 0) and plt.ylabel("GC activity", labelpad=0)
                
                ax.set_yticklabels([f"{y:g}" if y in yt else "" for y in yt])
                ax.set_xticklabels([f"{lab:g}" for lab in ax.get_xticks()])
                plt.xlabel("Time / sec.")            
            elif content == "RMSE":
                rmse = array([d["rmse"]/d["rmse"][0] for d in sweep[M,N,k]])
                INFO(f"  Plotting RMSE for {len(rmse)} trials.")
                mean_rmse = mean(rmse, axis=0)
                yl = (1e-3,2)
                ax,h = ft.plot0(tt - t_on, rmse.T,
                                col_cyc=cycler(color=[ft.set_alpha(0.25)([0.5,0.5,0.5,0.5])]),
                                xlim = xl, xticks=xt, ylim = yl, plot_fun = "semilogy",
                                plot_args={"linewidth":1})
                _,h = ft.plot0(tt - t_on, mean_rmse, ax = ax,
                               col_cyc=cycler(color=[cm_gc(0.75)]),
                               xlim = xl, xticks=xt, ylim = yl, plot_fun = "semilogy",
                               plot_args={"linewidth":1})
                
#                plt.semilogy(tt - t_on, mean_rmse, color=cm_gc(0.5))#data["rmse"]/data["rmse"][0])
                #ax.set_yticklabels([f"{log10(yt):g}" for yt in ax.get_yticks()])
                ax.set_xticklabels([f"{lab:g}".format(lab) for lab in ax.get_xticks()])
                (icol == 0) and plt.ylabel("RMS Error")
                plt.xlabel("Time / sec.")

    INFO(f"Finalizing figure.")
    plt.tight_layout()
    plt.gcf().align_ylabels()    
    labs = "ABCDEFGHIJKL"
    ft.label_axes([all_ax[r,0] for r in range(3)], labs, fontsize=14, verticalalignment="bottom", horizontalalignment="left",fontweight="bold")
    fig_file = "effect_of_size.pdf"
    INFO(f"Saving as {fig_file}.")    
    plt.savefig(fig_file, bbox_inches="tight")

    
    INFO("Finished {}.".format(inspect.stack()[0][3]))        
    
