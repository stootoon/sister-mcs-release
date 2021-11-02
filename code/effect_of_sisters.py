# This script contains the fucntions for loading the data for and plotting the effects of the number of sisters.

import datatools as dt
import logging
import numpy as np
from numpy import *
import util

from builtins import sum as bsum
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib import cycler
from matplotlib import ticker
import figtools as ft
import inspect
import pdb

logger = util.create_logger("effect_of_sisters")
INFO   = logger.info

rand = np.random.rand

def load_data(t_on = 0.1, t_decay = [0.4, 0.6], t_freq = [0.3, 0.6], kval = 3, ):
    INFO("Started {}.".format(inspect.stack()[0][3]))
    lev = logging.getLogger("olfactory_bulb").getEffectiveLevel()
    logging.getLogger("olfactory_bulb").setLevel(logging.WARNING)

    Svals = [1,2,8,25]

    unpack_results = lambda results: (results.results_array, results.t)
    
    data = {}
    for S in Svals:
        data[S], t = unpack_results(dt.load_Sk(S, k=kval))

    fs = int(round(1/(t[1] - t[0])))
    INFO(f"fs: {fs}")
        
    Svals1 = [1,2,4,8,12,16,20,25]
    decay = {"X":{}, "La":{}}
    F     = {"X":{}, "La":{}}
    rmse  = {"X":{}, "La":{}}

    # Assuming arr is in [timepoints x cells1 x cells2 x ...] format,
    # The next function pool the data for all cells at a given time point.    
    vectorize_cells = lambda arr: reshape(arr, (arr.shape[0], -1))
    
    for S in Svals1:
        data1, t = unpack_results(dt.load_Sk(S, k=kval))
        for fld in ["X","La"]:
            # Put the rmse data into time x trials format
            rmse[fld][S] = array([d[fld.lower()+"_rmse"] for d in data1]).T
            
            tt  = arange(rmse[fld][S].shape[0])/fs
            decay_slc = slice(int(t_decay[0]*fs),int(t_decay[1]*fs))
            INFO(f"Estimating decay using RMSE in the time slice {t_decay} (sec), which is {decay_slc}.")
            decay[fld][S] = [-1/polyfit(tt[decay_slc],log(rmse[fld][S][decay_slc,i]),1)[0] for i in range(rmse[fld][S].shape[1])]
            INFO("{:>2s} decay at S = {:2d}: {:.3f} +/- {:.3f}".format(fld, S, mean(decay[fld][S]), std(decay[fld][S])))

            freq_slc = slice(int(t_freq[0]*fs),int(t_freq[1]*fs))
            INFO(f"Estimating the fourier spectrum for the time slice {t_freq} (sec), which is {freq_slc}.")
            # The data is in [time_points x GCs|Glom x nsisters]
            # The reshape combines all cells for a single time point into a single vector            # 
            Z = array([vectorize_cells(d[fld]) for d in data1]).transpose([1,2,0]) # Times, cells, trials            
            f = fft.fft(Z[freq_slc],axis=0)
            nf = f.shape[0]
            fr = arange(nf)/nf*fs
            # Divide by nf so that e.g. a cosine will have a coefficient of 1/2
            F[fld][S] = mean(abs(reshape(f,(f.shape[0],-1))),axis=1)/nf;

    logging.getLogger("olfactory_bulb").setLevel(lev)
    INFO("Finished {}.".format(inspect.stack()[0][3]))    
    return {"data":data, "decay":decay, "F":F, "rmse":rmse, "fr":fr, "t":t}

def plot_effect_summary(data, ton = 0.1, seed = 1):
    # A combined plot showing some panels with the effect on MCs,
    # and another showing effect on GC RMSE.
    INFO("{} started.".format(inspect.stack()[0][3]))
    random.seed(seed)    
    data, decay, F, rmse, fr, t = [data[f] for f in ["data", "decay", "F", "rmse", "fr", "t"]]

    fs = int(round(1/(t[1] - t[0])))
    INFO(f"fs: {fs}")

    plt.figure(figsize=(8,6))
    ft.apply_styles()
    
    gs = GridSpec(2,2)
    
    cols_list, cols_dict = ft.Scols("mc")
    gc_cols_list, gc_cols_dict = ft.Scols("gc")
    
    # Set the glomerulus to show for each value of S
    # imc_tr_glom[S] = (trial, glom)
    imc_tr_glom = {2:[1,1], 8:[1,3], 25:[1,3]}  
    igc_tr = {1:1, 2:1, 8:1, 25:1} # Which trial to use for each S
    
    gax = {"spec":gs[0,0], # Amplitude spectrum
           "hfreq":gs[0,1], # Dependence of high-freq peak on S
           "tau":gs[1,0], # tau
           "rmse":gs[1,1], # RMSE
    }
    
    # PLOT TIME CONSTANTS FOR DIFFERENT S
    INFO("Plotting MC decay rates vs S.")    
    ss = array(sorted(decay["La"].keys()))
    mu = array([mean(array(decay["La"][s])) for s in ss])
    sd = array([std(array(decay["La"][s])) for s in ss])

    ax_decay = plt.subplot(gax["tau"])
    xt = [2,4,8,12,25]
    yt = [0, 50, 100]
    col_cyc = [cols_dict[S] for S in sorted(F["La"])]
    _, h = ft.plot0(ss.reshape((1,-1)), mu.reshape((1,-1))*1000,
                    plot_fun="plot", ax=ax_decay,
                    plot_args={"marker":"o","markersize":4},                    
                    col_cyc=cycler(color=col_cyc),
                    ylim=(0,120),
                    xticks = xt, xticklabels=[str(i) for i in xt],
                    yticks=yt,
                    #title="Decay rates", 
                    xlabel="S", ylabel="$\\tau$ (ms.) ", ylabel_args={"labelpad":-2})
    plt.legend(h, ["S = {}".format(s) for s in ss], prop={"size":8},frameon=False, labelspacing=0)

    ax_decay.tick_params(axis="x", which="minor", bottom=False)

    # Plot the AMPLITUDE SPECTRUM for different S
    INFO("Plotting MC amplitude spectra for different S.")
    col_cyc = [cols_dict[S] for S in sorted(F["La"])[1:]] # [1:] to skip S=1
    ax_freq = plt.subplot(gax["spec"])
    A = array([F["La"][S] for S in sorted(F["La"].keys()) if S>1]).T

    INFO("Peak amplitude: {:1.3e}".format(np.max(A[fr<500,:])))
    yt = [10**i for i in range(-3,0)]
    ft.plot0(fr[fr<500], A[fr<500,:],
             plot_fun = "loglog", ax=ax_freq,
             col_cyc = cycler(color=col_cyc), 
             xlim = (10,300), ylim=(2e-4,1e-1),
             yticks = yt, yticklabels = [f"{t:g}" for t in yt],
             xlabel = "Frequency (Hz)", xlabel_args = {"labelpad":-1},
             ylabel = "Amplitude",      ylabel_args = {"labelpad":-4},
             #title  = "Amplitude spectrum",
    )

    peak_freqs, peak_vals = dt.find_spectrum_peaks(A, fr)
    # Sometimes multiple peaks come up, so grab the highest one
    peak_freqs = array([pf[argmax(pv)] for pf, pv in zip(peak_freqs, peak_vals)])
    peak_vals  = array([max(pv) for pv in peak_vals])    
    for pf, pv in zip(peak_freqs, peak_vals):
        plt.plot(pf, pv, "rv", markersize=4)

    # Plot the PEAK FREQUENCY for different S
    INFO("Plotting square root fit to high frequency peak vs S.")
    ax_freq1 = plt.subplot(gax["hfreq"])

    ss = array([S for S in sorted(F["La"].keys()) if S>1])

    bb = mean(log(peak_freqs)) - 0.5*mean(log(ss))
    ff = 0.5*log(ss) + bb
    _, h = ft.plot0(log(ss), ff,
                    ax=ax_freq1,
                    col_cyc=cycler(color=["gray"]),
                    plot_args={"label":"$f \propto \sqrt{S}$", "linestyle":":"})
    ax_freq1.legend(prop={"size":8}, frameon=False, loc="lower right", labelspacing=0)
    ax_freq1.minorticks_off()
    #ax_freq1.set_xticks([])
    #ax_freq1.set_yticks([]) # ax_freq1.tick_params(axis="xy", which="minor", bottom=False, left=False)
    
    # Plot the HIGH FREQUENCY FIT
    INFO("Plotting MC high frequency peak vs S.")
    slab = [2,4,8,12,25]
    yt = [50*i for i in range(1,5)]
    yt_lab = [f"{t:g}" for t in yt]
    ft.plot0(log(ss.reshape((1,-1))), log(peak_freqs.reshape((1,-1))),
             #plot_fun = "loglog",
             ax = ax_freq1,
             col_cyc = cycler(color=col_cyc),             
             plot_args = {"marker":"o","markersize":6},             
#             ylim = (50,200), 
             xticks = log(slab), xticklabels = [str(s) if s in slab else "" for s in slab],             
             yticks = log(yt),   yticklabels = yt_lab,
             xlabel = "S",  ylabel = "Frequency (Hz)",
             #title = "High frequency mode",
    )

    # Plot the RMSE time courses
    INFO("Plotting GC RMSE time courses for different S.")    
    Svals = sorted(igc_tr.keys())
    errsn_all = {S:rmse["X"][S]/rmse["X"][S][0] for S in Svals}
    tt = np.arange(rmse["X"][Svals[0]].shape[0])/fs
    ax_rmse = plt.subplot(gax["rmse"])
    xl = (-ton,0.5)

    h = []

    # For aesthetics, plot S=1 at the bottom, but then from highest to lowest.
    Sorder = Svals[::-1]
    if Sorder[-1] == 1:
        Sorder = [1] + Sorder[:-1]

    h = {}
    for i, S in enumerate(Sorder):
        # offset = (10**(2*(i-len(Svals)+1)))
        offset = 1 # We don't want to stagger them
        yy = offset * errsn_all[S]
        my = mean(yy, axis=1)
        sy = std(yy,axis=1)
        color = gc_cols_dict[S]
        col_cyc = cycler(color=[color])
        ind_fit = np.where(((t-ton)>0.3) & ((t-ton)<0.5))[0]
        rmse_fit = np.polyfit((t-ton)[ind_fit], np.log(my[ind_fit]), 1)
        hf = None
        if S in [1,25]:
            mf = exp(rmse_fit[1] + rmse_fit[0]*(t-ton))
            hf = ax_rmse.semilogy(t - ton, mf*0.75, {1:":",25:"--"}[S], color=color, linewidth={1:1.5,25:1}[S],
                                  label=f"$\\tau$ = {-1000/rmse_fit[0]:2.0f} ms")
                            
        _,hi = ft.plot0(t - ton, my + 0*yy[:,1], ax=ax_rmse, col_cyc=col_cyc, plot_fun="semilogy",
                        xlim=xl, ylim=(1e-6,2), xlabel="Time (sec.)", plot_args={"linewidth":2, "label":f"S = {S}"})
        h[S] = [hi[0]]
        (hf is not None) and h[S].append(hf[0])

    ax_rmse.minorticks_off()    
    ax_rmse.yaxis.set_ticks([10**-i for i in range(0,8,2)])
    ax_rmse.set_ylabel("RMS Error", labelpad=-1)

    # Label the legend in numerical order
    h1 = bsum([[hi for hi in h[S]] for S in sorted(Sorder)], [])

    ax_rmse.legend(h1, [h.get_label() for h in h1], prop={"size":8}, frameon=False, loc="upper right", labelspacing=0, borderpad=0)
    
    plt.tight_layout(pad = 0, w_pad = 0.5, h_pad=0.5)
    ft.label_axes([ax_freq, ax_freq1, ax_decay, ax_rmse], "ABCDEF",fontsize=14,
                  verticalalignment="center", horizontalalignment="left",fontweight="bold")
    
    fig_file = "effect_of_S_summary.pdf"
    INFO(f"Saving as {fig_file}.")    
    plt.savefig(fig_file, bbox_inches="tight")

    INFO("{} finished.".format(inspect.stack()[0][3]))

    
def plot_effect_on_mcs(data, ton = 0.1, seed = 1):
    INFO("{} started.".format(inspect.stack()[0][3]))
    random.seed(seed)    
    data, decay, F, rmse, fr, t = [data[f] for f in ["data", "decay", "F", "rmse", "fr", "t"]]

    fs = int(round(1/(t[1] - t[0])))
    INFO(f"fs: {fs}")

    plt.figure(figsize=(8,6))
    ft.apply_styles()
    
    gs = GridSpec(5,6)
    
    cols_list, cols_dict = ft.Scols("mc")

    # Set the glomerulus to show for each value of S
    # imc_tr_glom[S] = (trial, glom)
    imc_tr_glom = {2:[1,1], 8:[1,3], 25:[1,3]}  

    gax = [gs[0,0:3],   # 2  sisters time course
           gs[1:3,0:3], # 8  sisters
           gs[:3,3:],   # 25 sisters
           gs[3:,:2],   # tau
           gs[3:,2:4],  # Amplitude spectrum
           gs[3:,4:]]   # Dependence of high-freq peak on S

    # PLOT THE TIME COURSES FOR DIFFERENT S
    INFO("Plotting MC time courses for different S.")
    ax = [plt.subplot(g) for g in gax[:3]]

    xl = (-0.01, 0.125)
    yl = {2:(-8,5), 8:(-15,6), 25:(-17,15)}
    for i, (S, (tr, glom)) in enumerate(imc_tr_glom.items()):
        INFO(f"Plotting MC time courses for S = {S}.")
        la  = data[S][tr]["La"][:,glom,:]
        ttl = "Mitral cell activity with 2 sisters" if S<=2 else f"{S} sisters"

        offsets = {
             2:zeros(la.shape),
             8:outer(ones(la.shape[0],), kron([-3.5,3.5],ones((4,)))),
            25:outer(ones(la.shape[0],), kron(arange(-2,3)*5, ones((5,))))
        }

        _ , h = ft.plot0(t - ton, la + offsets[S],
                         ax = ax[i], col_cyc = cycler(color=[ft.set_alpha(0.5 if S==25 else 0.75)(cols_dict[S])]),
                         xlim = xl,
                         xticks = [0,0.05,0.1],
        )

        # Plot the y-scale bar
        xl = plt.xlim()
        ax[i].plot([xl[0], xl[0]], [yl[S][0], yl[S][0]+5],"k", linewidth=2)
        ax[i].spines['left'].set_visible(False)
        ax[i].set_yticks([])
        ax[i].set_ylim(yl[S])
        ax[i].set_xlim(xl)

        if S in [8,25]:
            plt.xlabel("Time (sec.)", labelpad=0)
    
    ax[0].yaxis.set_label_coords(-0.3,0.5)
    
    
    # PLOT TIME CONSTANTS FOR DIFFERENT S
    INFO("Plotting MC decay rates vs S.")    
    ss = array(sorted(decay["La"].keys()))
    mu = array([mean(array(decay["La"][s])) for s in ss])
    sd = array([std(array(decay["La"][s])) for s in ss])

    ax_decay = plt.subplot(gax[3])
    xt = [2,4,8,12,25]
    yt = [0, 50, 100]
    col_cyc = [cols_dict[S] for S in sorted(F["La"])]
    _, h = ft.plot0(ss.reshape((1,-1)), mu.reshape((1,-1))*1000,
                    plot_fun="plot", ax=ax_decay,
                    plot_args={"marker":"o","markersize":4},                    
                    col_cyc=cycler(color=col_cyc),
                    ylim=(0,120),
                    xticks = xt, xticklabels=[str(i) for i in xt],
                    yticks=yt,
                    #title="Decay rates", 
                    xlabel="S", ylabel="$\\tau$ (ms.) ", ylabel_args={"labelpad":-2})
    plt.legend(h, ["S = {}".format(s) for s in ss], prop={"size":8},frameon=False, labelspacing=0)

    ax_decay.tick_params(axis="x", which="minor", bottom=False)

    # Plot the AMPLITUDE SPECTRUM for different S
    INFO("Plotting MC amplitude spectra for different S.")
    col_cyc = [cols_dict[S] for S in sorted(F["La"])[1:]] # [1:] to skip S=1
    ax_freq = plt.subplot(gax[4])
    A = array([F["La"][S] for S in sorted(F["La"].keys()) if S>1]).T

    INFO("Peak amplitude: {:1.3e}".format(np.max(A[fr<500,:])))
    yt = [10**i for i in range(-3,0)]
    ft.plot0(fr[fr<500], A[fr<500,:],
             plot_fun = "loglog", ax=ax_freq,
             col_cyc = cycler(color=col_cyc), 
             xlim = (10,300), ylim=(2e-4,1e-1),
             yticks = yt, yticklabels = [f"{t:g}" for t in yt],
             xlabel = "Frequency (Hz)", xlabel_args = {"labelpad":-1},
             ylabel = "Amplitude",      ylabel_args = {"labelpad":-4},
             #title  = "Amplitude spectrum",
    )

    peak_freqs, peak_vals = dt.find_spectrum_peaks(A, fr)
    # Sometimes multiple peaks come up, so grab the highest one
    peak_freqs = array([pf[argmax(pv)] for pf, pv in zip(peak_freqs, peak_vals)])
    peak_vals  = array([max(pv) for pv in peak_vals])    
    for pf, pv in zip(peak_freqs, peak_vals):
        plt.plot(pf, pv, "rv", markersize=4)

    # Plot the PEAK FREQUENCY for different S
    INFO("Plotting square root fit to high frequency peak vs S.")
    ax_freq1 = plt.subplot(gax[5])

    ss = array([S for S in sorted(F["La"].keys()) if S>1])

    bb = mean(log(peak_freqs)) - 0.5*mean(log(ss))
    ff = 0.5*log(ss) + bb
    _, h = ft.plot0(ss, exp(ff),
                    ax=ax_freq1,
                    col_cyc=cycler(color=["gray"]),
                    plot_args={"label":"$f \propto \sqrt{S}$", "linestyle":":"})

    # Plot the HIGH FREQUENCY FIT
    INFO("Plotting MC high frequency peak vs S.")    
    slab = [2,4,8,12,25]
    yt = [50*i for i in range(1,5)]
    yt_lab = [f"{t:g}" for t in yt]
    ft.plot0(ss.reshape((1,-1)), peak_freqs.reshape((1,-1)),
             plot_fun = "loglog", ax = ax_freq1,
             col_cyc = cycler(color=col_cyc),             
             plot_args = {"marker":"o","markersize":6},             
#             ylim = (50,200), 
             xticks = slab, xticklabels = [str(s) if s in slab else "" for s in slab],             
             yticks = yt,   yticklabels = yt_lab,
             xlabel = "S",  ylabel = "Frequency (Hz)",
             #title = "High frequency mode",
    )
    
    ax_freq1.legend(prop={"size":8}, frameon=False, loc="lower right", labelspacing=0)
    ax_freq1.minorticks_off() # ax_freq1.tick_params(axis="xy", which="minor", bottom=False, left=False)
    plt.tight_layout(pad = 0, w_pad = -0.5, h_pad=0.5)
    ft.label_axes(ax + [ax_decay,ax_freq,ax_freq1], "ABCDEF",fontsize=14,
                  verticalalignment="center", horizontalalignment="left",fontweight="bold")
    
    l,b,w,h = ax[0].get_position().bounds
    ax[0].set_position([l,b-0.035,w,h+0.035])
    fig_file = "effect_of_S_on_MCs.pdf"
    INFO(f"Saving as {fig_file}.")    
    plt.savefig(fig_file, bbox_inches="tight")
    INFO("{} finished.".format(inspect.stack()[0][3]))
    
def plot_effect_on_gcs(data, ton = 0.1, seed = 1):
    this_fun = inspect.stack()[0][3]
    INFO("{} started.".format(this_fun))
    random.seed(seed)    
    data, decay, F, rmse, fr, t = [data[f] for f in ["data", "decay", "F", "rmse", "fr", "t"]]
    fs = int(round(1/(t[1] - t[0])))
    INFO(f"fs: {fs}")  

    cols_list, cols_dict = ft.Scols("gc")
    
    igc_tr = {1:1, 2:1, 8:1, 25:1} # Which trial to use for each S

    plt.figure(figsize=(8,7))
    ft.apply_styles()
    
    gs  = GridSpec(8,8)
    gax = [gs[0:2,0:4], gs[0:2,4:], gs[2:4,0:4], gs[2:4,4:]]

    INFO("Plotting GC time courses for different S.")
    ax = [plt.subplot(g) for g in gax[:4]]
    xl = [-0.01,0.25]
    for i, (S, tr) in enumerate(igc_tr.items()):
        INFO(f"Plotting GC time courses for S = {S}.")
        x     = data[S][tr]["X"]
        x_MAP = data[S][tr]["x_MAP"]
        i_MAP = [i for i in range(len(x_MAP))   if x_MAP[i] > 0.5]            
        i_out = [i for i in np.argsort(x[-1,:]) if i not in i_MAP]

        if S==1:
            ttl = "1 MC / glomerulus"
        elif S == 2:
            ttl = "2 sisters / glomerulus"
        else:
            ttl = "{} sisters".format(S)

        col_cyc = cycler(color=ft.gen_gc_cols(central_color = cols_dict[S]))
        _, h = ft.plot0(t - ton, x,
                        ax = ax[i], col_cyc = col_cyc,
                        xlim = xl, xticks = [0,0.1,0.2],
                        ylim=(-0.1,2.1), yticks=[0,1,2],
                        xlabel="Time / sec.",
                        ylabel = "GC activity",)
        [plt.plot(ax[i].get_xlim()[1], x, "r<", linewidth=1) for x in x_MAP[x_MAP>0.5]];



    INFO("Plotting GC RMSE time courses for different S.")    
    Svals = sorted(igc_tr.keys())
    errsn_all = {S:rmse["X"][S]/rmse["X"][S][0] for S in Svals}
    tt = np.arange(rmse["X"][S].shape[0])/fs
    ax_rmse = plt.subplot(gs[4:,0:2])
    xl = (-ton,0.5)
    labs = []
    h = []
    for i, S in enumerate(Svals):
        offset = (10**(2*(i-len(Svals)+1)))
        yy = offset * errsn_all[S]
        my = mean(yy, axis=1)
        sy = std(yy,axis=1)
        color = cols_dict[S]
        col_cyc = cycler(color=[color])
        _,hi = ft.plot0(t - ton, yy[:,1], ax=ax_rmse, col_cyc=col_cyc, plot_fun="semilogy",
                        xlim=xl, ylim=(1e-10,2), xlabel="Time (sec.)", plot_args={"linewidth":1, "label":f"S = {S}"},
        )
        ft.plot0(t - ton, my, ax=ax_rmse, col_cyc=cycler(color=["r"]), plot_fun="semilogy", xlim=xl, ylim=(1e-10,2), xlabel="Time (sec.)")
        
    ax_rmse.yaxis.set_ticks([10**-i for i in range(0,12,2)])
    ax_rmse.set_ylabel("RMS Error", labelpad=-5)
    plt.legend(prop={"size":8}, frameon=False, loc="upper right", labelspacing=0, borderpad=0)
    
    INFO("Plotting GC decay rates vs S.")    
    ss = array(sorted(decay["X"].keys()))
    mu = array([mean(array(decay["X"][s])) for s in ss])
    sd = array([std(array(decay["X"][s])) for s in ss])

    col_cyc = [cols_dict[S] for S in sorted(F["La"])]    
    ax_decay = plt.subplot(gs[4:,2:4])
    xt = [2,12,25]    
    _, h = ft.plot0(ss.reshape((1,-1)), mu.reshape((1,-1)), ax=ax_decay, col_cyc=cycler(color=col_cyc), plot_args={"marker":"o","markersize":4})
    plt.legend(h, ["S = {}".format(s) for s in ss], prop={"size":8},frameon=False, loc="lower right", labelspacing=0)
    ft.plot0(array([ss,ss]), array([mu-sd,mu+sd]),
             xticks = xt, ax=ax_decay,
             col_cyc=cycler(color=col_cyc),
             plot_fun="plot", xlabel="S",
             # title="Decay rates",
             ylim=(0,0.125))
    plt.ylabel("$\\tau$ (sec.)", labelpad=0)
    ax_decay.tick_params(axis="x", which="minor", bottom=False)

    INFO("Plotting GC amplitude spectra for different S.")
    col_cyc = [cols_dict[S] for S in sorted(F["X"])[1:]] # [1:] to skip S=1    
    ax_freq = plt.subplot(gs[4:,4:6])
    A    = np.array([F["X"][S] for S in sorted(F["X"]) if S>1]).T
    
    ft.plot0(fr[fr<500], A[fr<500,:],
             ax=ax_freq, col_cyc=cycler(color=col_cyc),
             plot_fun="loglog",
             xlim=(20,400), ylim=(1e-7,1e-5),
             xlabel = "Frequency (Hz)",
             #title="Amp. spectrum"
    )

    peak_freqs, peak_vals = dt.find_spectrum_peaks(A, fr, min_freq = 50, width=5)
    # Sometimes multiple peaks come up, so grab the highest one
    peak_freqs = array([pf[argmax(pv)] for pf, pv in zip(peak_freqs, peak_vals)])
    peak_vals  = array([max(pv) for pv in peak_vals])        
    for pf, pv in zip(peak_freqs, peak_vals):
        plt.plot(pf, pv,"rv", markersize=4)

    INFO("Plotting fit to high frequency peak.")    
    ax_freq1   = plt.subplot(gs[4:,6:])
    ss = array([S for S in sorted(F["X"].keys()) if S>1])
    bb = mean(log(peak_freqs)) - 0.5*mean(log(ss))
    ff = 0.5*log(ss) + bb
    ft.plot0(ss, exp(ff),
             ax=ax_freq1, col_cyc=cycler(color=["lightgray"]),
             plot_args={"label":"$f \propto \sqrt{S}$"})
    ft.plot0(ss.reshape((1,-1)), peak_freqs.reshape((1,-1)),
             ax=ax_freq1, col_cyc=cycler(color=col_cyc),
             plot_args={"marker":"o","markersize":6},
             plot_fun = "loglog", xlabel="S",
             # title="High freq. mode"
    )
    ax_freq1.legend(prop={"size":8},frameon=False, loc="lower right", labelspacing=0)

    slab = [2,4,8,12,25]
    ax_freq1.set_xticks(slab)
    ax_freq1.set_xticklabels([str(s) if s in slab else "" for s in slab])
    yt = arange(50,300,50)
    ax_freq1.set_yticks(yt)
    ax_freq1.set_yticklabels(["{:d}".format(yi) for yi in yt])
    ax_freq1.minorticks_off()

    INFO(f"Finalizing figure.")
    plt.tight_layout()
    
    l,b,w,h = ax_rmse.get_position().bounds
    ax_rmse.set_position([l,b,w*1.125,h])
    
    l,b,w,h = ax_decay.get_position().bounds
    
    l,b,w,h = ax_freq.get_position().bounds
    ax_freq.set_position([l,b,w*1.125,h])
    ax_freq.yaxis.set_label_coords(-0.250,0.5)
    ax_freq.set_ylabel("Amplitude", labelpad=0)
    
    l,b,w,h = ax_freq1.get_position().bounds
    ax_freq1.set_position([l,b,w*1.1,h])
    ax_freq1.yaxis.set_label_coords(-0.205,0.5)    
    ax_freq1.set_ylabel("Frequency (Hz)", labelpad=-20)
    
    ft.label_axes(ax + [ax_rmse, ax_decay,ax_freq,ax_freq1], "ABCDEFGH",fontsize=14, verticalalignment="center", horizontalalignment="left",fontweight="bold")

    fig_file = "effect_of_S_on_GCs.pdf"
    INFO(f"Saving as {fig_file}.")    
    plt.savefig(fig_file, bbox_inches="tight")                

    INFO("{} finished.".format(this_fun))

