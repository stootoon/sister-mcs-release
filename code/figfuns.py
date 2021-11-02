import os
opj = os.path.join
SISTER_MCS = os.environ["SISTER_MCS"]
import pickle
from matplotlib.gridspec import GridSpec
from matplotlib import cycler
from matplotlib import pyplot as plt
from numpy import *
import numpy as np
from matplotlib import cm
import figtools  as ft
import datatools as dt
import util
import inspect
logger = util.create_logger("figfuns")
INFO = logger.info

rand = random.rand

def load_data(S, leak_pg, k = 3, which_seed = None, vars = None, sweep_folder = "sweep_conc_spread"):
    data_folder  = dt.data_folder(sweep_folder)
    df, _        = dt.load_params_from_folder(data_folder)

    selector = (np.abs(df.leak_pg - leak_pg) <1e-8) & (df.S == S) & (df.k == k)
    seeds = df[selector]["seed"].tolist()
    INFO(f"Found {len(seeds)} seeds for {S=} and {leak_pg=} and {k=}.")
    
    if which_seed is None:
        which_seed = [min(seeds)]
    elif which_seed == "all":
        which_seed = sorted(seeds)
    elif which_seed not in seeds:
        raise ValueError(f"Requested seed {seed} is not available for {S=} and {leak_pg=} and {k=}.")

    if len(which_seed) == 0:
        raise ValueError(f"No seeds found for {S=} and {leak_pg=} and {k=}.")
    
    seed_mask = (df.seed == which_seed[0])
    for seed in which_seed[1:]:
        seed_mask |= (df.seed == seed)
    df_sub = df[(selector) & (seed_mask)]
    if len(df_sub) != len(which_seed):
        raise ValueError(f"Expected exactly {len(which_seed)} parameter file for {S=} and {k=} and {leak_pg=} but found {len(df_sub)}.")

    params_files = df_sub["file"].tolist()
    params_dirs  = [str(pf.strip(".json")) for pf in params_files]
    results = [dt.load_results(data_folder, pd, vars=vars) for pd in params_dirs]
    results = dt._post_load(results, params_files, data_folder) # Add rmse etc...

    return results[0] if len(which_seed) == 1 else {seed:res for (seed,res) in zip(which_seed, results)}

def plot_glom1_glom2(leak = 0, S = 4, imc = [2,7], figsize=(7,8), xl = (-0.01,0.5), gs = None, ylabels=True, label_axes = True, save_figure = True, plot_args = {"linewidth":2}):
    INFO("Started {}.".format(inspect.stack()[0][3]))
    
    results = load_data(S, leak)
    x_MAP   = results["x_MAP"]

    cm_mc = ft.pop_cmaps["mc"]
    cm_pg = ft.pop_cmaps["pg"]
    cm_gc = ft.pop_cmaps["gc"]

    # Panel 1: 4 sisters from the first glomerulus
    ton = 0.1

    random.seed(1)
    if gs is None:
        plt.figure(figsize=figsize)
        gs = GridSpec(3+len(imc),1)
        ft.apply_styles()
    
    n_mc = results["La"].shape[-1]

    cyc_mc = cycler(color=ft.gen_hue_spread_colors(cm_mc(0.5), S, alpha=0.75))
    cyc_pg = cycler(color=ft.gen_hue_spread_colors(cm_pg(0.5), S, alpha=0.75))

    yt = array([-5,0,5,10])
    yt_pg = [-1,-0.5,0,0.5]
    xt = arange(0,0.501,0.1)

    yl_mc = [-7,7]
    yl_pg = (-0.59, 0.59)

    ax_mc = []
    for i in range(len(imc)):
        INFO(f"Plotting mitral cells in glomerulus {imc[i]}.")
        axi, h = ft.plot0(results["T"] - ton, 0*results["T"], ax=plt.subplot(gs[i]), col_cyc="lightgray", plot_args={"linewidth":1})    
        axi, h = ft.plot0(results["T"] - ton, results["La"][:,imc[i],:], ax = axi, col_cyc = cyc_mc, yticks = yt, xticks = xt, ylim = yl_mc, xlim = xl, xlabel = "Time (sec.)", ylabel = "MC activity" if ylabels else None, plot_args = plot_args);
        ax_mc.append(axi)

    n_mc = len(ax_mc)
    INFO(f"Plotting periglomerular cells in glomerulus {imc[0]}.")    
    ax5, h = ft.plot0(results["T"] - ton, 0*results["T"], ax =plt.subplot(gs[n_mc]), col_cyc="lightgray", plot_args={"linewidth":1}, xlabel="Time (sec.)")
    ax5, h = ft.plot0(results["T"] - ton, results["Mu"][:,imc[0],:], ax = ax5, col_cyc = cyc_pg, ylim = yl_pg, yticks = yt_pg, xticks = xt, xlim = xl, xlabel = "Time (sec.)", ylabel="PG activity" if ylabels else None, plot_args=plot_args);

    INFO(f"Plotting granule cells.")        
    ax6, h = ft.plot0(results["T"] - ton, results["X"], ax = plt.subplot(gs[n_mc+1]), col_cyc = ft.cyc_gc, plot_args = plot_args, ylim = (-0.1, 3), xlim = xl, xticks = xt, xlabel="Time (sec.)", ylabel  = "GC activity" if ylabels else None, yticks = [0,1,2], yticklabels=["0","1","2"])
    [plt.plot(plt.xlim()[1], x, "r<", linewidth=1) for x in x_MAP[x_MAP>0.5]];

    INFO(f"Plotting RMSE.")            
    ax7, h = ft.plot0(results["T"] - ton, results["rmse"]/results["rmse"][0], ax = plt.subplot(gs[n_mc+2]), col_cyc = cycler(color=[cm_gc(0.9)]), plot_args = plot_args,
                      plot_fun = "semilogy", xlim=xl, xticks = xt, ylim=(1e-4 if leak is False else 1e-3,2), xlabel="Time (sec.)", ylabel="RMS error" if ylabels else None)

    INFO(f"Finalizing figure.")
    plt.tight_layout(pad = 0, h_pad = 0)
    labs = "ABCDEFGH"
    ax = ax_mc + [ax5,ax6,ax7]

    if label_axes is True:
        ft.label_axes(ax, labs, fontsize=14, verticalalignment="top", horizontalalignment="left",fontweight="bold")

    if save_figure is True:
        file_name = "glom1_glom2_S{}{}.pdf".format(S, "_leak" if leak is True else "")
        INFO(f"Writing {file_name}.")
        plt.savefig(file_name, bbox_inches="tight")

    INFO("Finished {}.".format(inspect.stack()[0][3]))
    return ax
