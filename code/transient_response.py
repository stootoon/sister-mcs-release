from numpy import *
import numpy as np
import effect_of_sisters as eos
import effect_of_parameters as eop
import datatools as dt
import figtools as ft
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cycler
from matplotlib import ticker
import figtools as ft
import inspect
import util

logger = util.create_logger("transient_response")
INFO   = logger.info

def drop_dict_fields(d, fields):
    for f in fields:
        d.pop(f, None)
    return d

def drop_sweep_data(swp):
    del swp._data
    return swp

def load_data(eos_args={}, eop_args={}, force_compute = False):
    return {"eos": drop_dict_fields(dt.load_if_exists_else_compute("eos_data.p", lambda: eos.load_data(**eos_args), force_compute = force_compute), ["data"]),
            "eop": drop_sweep_data(dt.load_if_exists_else_compute("eop_data.p", lambda: eop.find_amplitude_spectrum_peaks(eop.load_data(**eop_args)), force_compute = force_compute))}

def plot_transient_response(data, color=cm.Blues(0.65)):
    decay, F, rmse, fr, t = [data["eos"][f] for f in ["decay", "F", "rmse", "fr", "t"]]

    plt.figure(figsize=(8,3))
    ft.apply_styles()

    gs = GridSpec(1,3)
    gax = {"tau_vs_S":gs[0,2], # Amplitude spectrum
           "hfreq_vs_S":gs[0,0], # Dependence of high-freq peak on S
           "hfreq_vs_sd":gs[0,1], # Dependence of high-freq peak on sd
    }
    
    cols_list, cols_dict = ft.Scols("mc")
    
    # PLOT TIME CONSTANTS FOR DIFFERENT S
    INFO("Plotting MC decay rates vs S.")    

    ss = array(sorted(decay["La"].keys()))
    mu = array([mean(array(decay["La"][s])) for s in ss])
    sd = array([std(array(decay["La"][s]))  for s in ss])

    ax_decay = plt.subplot(gax["tau_vs_S"])
    xt = [1,2,4,8,12,25]
    yt = [0, 50, 100]    
    col_cyc = [cols_dict[S] for S in sorted(F["La"])]
    _, h = ft.plot0(ss.reshape((1,-1)), mu.reshape((1,-1))*1000,
                    plot_fun="plot", ax=ax_decay,
                    plot_args={"marker":"o","markersize":4},                    
                    col_cyc=cycler(color=col_cyc),
                    ylim=(0,120),
                    xticks = ss, xticklabels=[str(i) for i in ss],
                    yticks=yt,
                    #title="Decay rates", 
                    xlabel="S", ylabel="$\\tau$ (ms.) ", ylabel_args={"labelpad":-2})

    ax_decay.tick_params(axis="x", which="minor", bottom=False)


    def _plot_dependence(ax, ss, peak_freqs, c, x_lab, x_tex,
                         xt = [2,4,8,12,25], xtf = str, yt = [50*i for i in range(1,5)],
                         cols = [cols_dict[8]],
                         leg_loc="best"):
        # First plot the fit
        bb = mean(log(peak_freqs)) - c*mean(log(ss))
        ff = c*log(ss) + bb
        _, h = ft.plot0(log(ss), ff,
                        ax=ax,
                        col_cyc=cycler(color=["gray"]),
                        plot_args={"label":"$f \propto " + x_tex + "$", "linestyle":":"})
        ax.legend(prop={"size":8}, loc=leg_loc, frameon=False, labelspacing=0)
        ax.minorticks_off()
    
        # Then plot the data on top
        yt_lab = [f"{t:g}" for t in yt]
        ft.plot0(log(ss.reshape((1,-1))), log(peak_freqs.reshape((1,-1))),
                 ax = ax,
                 col_cyc = cycler(color=cols),             
                 plot_args = {"marker":"o","markersize":6},             
                 xticks = log(xt), xticklabels = [xtf(s) for s in xt],             
                 yticks = log(yt), yticklabels = yt_lab,
                 xlabel = x_lab,  ylabel = "Frequency (Hz)",
        )

        
    # Sometimes multiple peaks come up, so grab the highest one
    A = array([F["La"][S] for S in sorted(F["La"].keys()) if S>1]).T
    peak_freqs, peak_vals = dt.find_spectrum_peaks(A, fr)
    peak_freqs = array([pf[argmax(pv)] for pf, pv in zip(peak_freqs, peak_vals)])
    peak_vals  = array([max(pv) for pv in peak_vals])    
    ss = array([S for S in sorted(F["La"].keys()) if S>1])
    INFO("Plotting square root fit to high frequency peak vs S.")
    ax_freq1 = plt.subplot(gax["hfreq_vs_S"])
    _plot_dependence(ax_freq1,
                     ss, peak_freqs, 0.5, "S", "\\sqrt{S}",
                     leg_loc="lower right",
                     cols = col_cyc[1:]
    )
    yl = log(array([30,200]))
    ax_freq1.set_ylim(yl)
    # subplot(1,3,3)
    eopd = data["eop"]
    param_vals = array(eopd.get_param_vals("sd"))
    pf = array([eopd.peak_freqs["sd", pv, "La"][:2] for pv in param_vals])
    pv = array([eopd.peak_vals["sd", pv, "La"][:2]  for pv in param_vals])

    INFO("Plotting linear fit to high frequency peak vs σ.")
    ax_freq2 = plt.subplot(gax["hfreq_vs_sd"])
    _plot_dependence(ax_freq2,
                     param_vals, pf[:,-1], -1,
                     "$\sigma$", "\\sigma^{-1}",
                     xt  = sorted(param_vals),
                     xtf = lambda x: str(x) if len(str(x))<=4 else "",
                     leg_loc = "lower left",
                     cols = [cols_dict[4]], # The eop data is for S = 4
    )
    ax_freq2.set_ylim(yl)

    plt.tight_layout(pad = 0, w_pad = -0, h_pad=0)
    ft.label_axes([ax_freq1, ax_freq2,ax_decay], "ABC",fontsize=14,
                  verticalalignment="center", horizontalalignment="left",fontweight="bold")
    fig_file = f"transient_response.pdf"
    logger.info(f"Saving as {fig_file}.")    
    plt.savefig(fig_file, bbox_inches="tight")
    logger.info("{} finished.".format(inspect.stack()[0][3]))        
        
