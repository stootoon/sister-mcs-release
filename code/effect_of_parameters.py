# This script contains the fucntions for loading the data for and plotting the effects of the time constants and prior parameters
import os
import logging
import numpy as np
from numpy import *
import util
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib import cycler
from matplotlib import ticker
import figtools as ft
import datatools as dt
import inspect

import olfactory_bulb as ob

logger = util.create_logger("effect_of_params")

plt.style.use("default")
plt.style.use("./ms.mplstyle")

rand = np.random.rand

def load_data(params = ["tau_pg", "tau_mc", "tau_gc", "sd", "be", "ga"], S = 4, t_decay = [0.4, 0.6], t_freq = [0.3, 0.6], kval=3, n_oversample_spectrum=1, **kwargs):
    """
    RETURNS: A SweepDataset object containing the data for the specified parameters,
    and additional fields 'fr', 'ix', and 't_decay'.
    """
    logger.info("Started {}.".format(inspect.stack()[0][3]))
    lev = logging.getLogger("olfactory_bulb").getEffectiveLevel()
    logging.getLogger("olfactory_bulb").setLevel(logging.WARNING)

    sweep_k3 = dt.SweepDataset(os.path.join(os.environ["SISTER_MCS_DATA"], "sweep_all_but_k3"))
    sweep_k3.load_sweep_for_parameters(S, params, **kwargs)
    
    t  = sweep_k3.get_time()
    logger.info("{} time indices from {:1.3f} - {:1.3f} seconds.".format(len(t), t[0], t[-1]))
    
    fs = int(round(1/(t[1] - t[0])))
    logger.info(f"fs: {fs}")    

    N = sweep_k3.get_instance_of_field("X").shape[1]
    logger.info(f"# GCs: {N}")

    sweep_k3.ix = where(ob.get_x_true(N, kval))[0]
    logger.info(f"Non zero indices in true odour vector: {sweep_k3.ix}")

    # We will now need to compute some functions that average across trials, glomeruli and sisters.
    # Hence we will need to reorder the data so that the time axis comes first.
    time_first = lambda X: X.transpose([1,0] + list(range(2,len(X.shape))))    
    # We will then need to flatten the data into time x [columns] matrix
    flatten = lambda X: X.reshape((X.shape[0], -1))
    # Combine these to rearrange the data
    rearrange = lambda X: flatten(time_first(X))

    logger.info("Computing amplitude spectra.")
    freq_slc = slice(int(t_freq[0]*fs),int(t_freq[1]*fs))
    logger.info("Using time window {:1.3f} - {:1.3f} sec. (samples {} - {}).".format(t_freq[0], t_freq[1], freq_slc.start, freq_slc.stop))    
    freq_sub = lambda X: X[freq_slc]
    n_freq   = freq_slc.stop  - freq_slc.start
    n_fft    = n_freq * n_oversample_spectrum
    fr       = arange(n_fft)/n_fft*fs
    fr_mask  = (fr > 5) & (fr<500)

    sweep_k3.fr = fr
    
    amplitude_spectrum = lambda X: mean(abs(fft.fft(freq_sub(rearrange(X)),axis=0, n = n_fft)),axis=1)
    sweep_k3.apply("amplitude_spectrum", amplitude_spectrum, ["La", "Mu", "X"])

    logger.info("Computing decay time constants.")
    decay_slc = slice(int(t_decay[0]*fs),int(t_decay[1]*fs))
    logger.info("Using time window {:1.3f} - {:1.3f} sec. (samples {} - {}).".format(t_decay[0], t_decay[1], decay_slc.start, decay_slc.stop))    
    decay_sub = lambda X: X[decay_slc]
    n_decay   = decay_slc.stop - decay_slc.start
    tdecay    = arange(n_decay)/fs

    sweep_k3.t_decay = tdecay + t_decay[0]
    
    #D = {pop:{param:array([rmsf(flatten(merge(data[p], pop)))[decay_slc] for p in sorted(data.keys())]).T for param,data in dd.items()} for pop in ["La", "Mu", "X"]}
    #decays = {pop:{param:polyfit(tdecay, log(data), 1)[0] for param, data in Dpop.items()} for pop, Dpop in D.items()}
    mean_rmse = lambda X: decay_sub(mean(X, axis=0)) # Average the rmse across trials    
    sweep_k3.apply("mean_rmse", mean_rmse, ["la_rmse","mu_rmse","x_rmse"])

    # We don't save these any more because
    # (a) we don't use them later, and
    # (b) they can't be pickled.

    #sweep_k3.mean_rmse_fun = mean_rmse
    
    decay_tau = lambda X: -1/polyfit(tdecay, log(mean_rmse(X)), 1)[0]
    sweep_k3.apply("decay_tau", decay_tau, ["la_rmse","mu_rmse","x_rmse"])
    #sweep_k3.decay_tau_fun = decay_tau
    
    logging.getLogger("olfactory_bulb").setLevel(lev)    
    logger.info("Finished {}.".format(inspect.stack()[0][3]))    
    return sweep_k3


def find_amplitude_spectrum_peaks(sweep, fields = ["La", "Mu", "X"]):
    # Used to augment the results of a load_data call with the location of the frequency peaks
    fr = sweep.fr
    orders = {"Mu":4, "La":4, "X":8} # For find_peaks using argrelextrema
    sweep.__dict__["peak_freqs"] = {}
    sweep.__dict__["peak_vals"]  = {}
    logger.info(f"Finding amplitude spectrum peaks for fields {fields}")
    for ip, param in enumerate(sweep.get_loaded_params()):
        logger.info(f"Finding amplitude spectrum peaks for parameter '{param}'.")        
        param_vals = sorted(sweep.get_param_vals(param))
        for fld in fields:
            A = np.array([sweep.amplitude_spectrum[param, pv, fld][fr>5] for pv in param_vals]).T
            peak_freqs, peak_vals = dt.find_spectrum_peaks(A, fr[fr>5], min_freq=10, method="argrelextrema", order = orders[fld])
            for iv, pv in enumerate(param_vals):
                sweep.peak_freqs[param, pv, fld] = peak_freqs[iv]
                sweep.peak_vals[param,  pv, fld] = peak_vals[iv]
    return sweep

def plot_effect_on_population(sweep, ton = 0.1, seed = 1, pop="mc"):
    logger.info("{} started.".format(inspect.stack()[0][3]))
    
    pop = pop.lower()
    if pop not in ["mc", "gc", "pg"]:
        raise ValueError(f"'pop' must be one of 'mc', 'gc', or 'pg', was 'f{pop}'.")
    

    random.seed(seed)

    pop_tr = {"mc":"La", "pg":"Mu", "gc":"X"} # Translate population to data fields
    fld    = pop_tr[pop]
    fld_rmse = fld.lower() + "_rmse"
    ttls   = {"tau_mc": "$\\tau_\\lambda$", "tau_pg": "$\\tau_\\mu$", "tau_gc":"$\\tau_v$", "sd":"$\\sigma$", "ga":"$\\gamma$", "be":"$\\beta$"}


    t  = sweep.get_time()
    fs = int(round(1/(t[1] - t[0])))
    logger.info(f"fs: {fs}")

    ix = sweep.ix
    fr = sweep.fr

    loaded_params = sweep.get_loaded_params()
    n_params      = len(sweep.get_param_vals(loaded_params[0]))

    cols_list, cols_dict = ft.Scols(pop)
    # Color for the central parameter values
    center_color = cols_dict[sweep.S]    
    # Colors for other example traces in the background
    gray_colors  = [ft.set_alpha(0.75)(ft.cm.gray(rand()*0.25+0.75)) for i in range(100)]
    #param_colors = [ft.set_alpha(0.75)(ft.color_for_param(center_color, ip, start = 1/4, end =1)) for ip in np.linspace(-1,1, n_params)]
    param_colors = ft.gen_hue_spread_colors(center_color, n_params, hue_spread = 0.2, alpha = 1)    
    param_colors = [ft.set_alpha(0.75)(cols_dict[S]) for S in [5,10,15,20,25]]
    
    plt.figure(figsize=(13.5,16))
    ft.apply_styles()
    
    gs = GridSpec(14,6)
    which_trial_to_plot = 0
    for ip, param in enumerate(loaded_params):
        ttl = ttls[param]
        param_vals = sorted(sweep.get_param_vals(param))
        ymin, ymax = [inf, -inf]
        row = 2*ip
        ax = []        
        for iv, pv in enumerate(param_vals[::2]):
            ax.append(plt.subplot(gs[row:row+2,iv]))
            X = copy(sweep[param, pv, fld][which_trial_to_plot]);
            X = X[:, ix] if pop == "gc" else X[:,0,:] # Plot the active GCs, or the sisters from the first glomerulus
            nx = X.shape[1]
            colors_list = gray_colors[:nx-1] + [param_colors[param_vals.index(pv)]] 
            col_cyc = cycler(color=colors_list)
            ft.plot0(t - ton, X, ax=ax[-1], col_cyc = col_cyc, xlabel =  "Time (sec.)", title=f"{ttls[param]}: {pv:g}",
                     xlim=(-0.01,0.5) if pop != "gc" else (-0.01,0.25))
            ymin = min(ymin, plt.ylim()[0])
            ymax = max(ymax, plt.ylim()[1])

        # Ploting the amplitude spectra
        logger.info(f"Plotting amplitude spectra for '{param}'.")
        ax = plt.subplot(gs[row:row+2,3])
        A = np.array([sweep.amplitude_spectrum[param, pv, fld][fr>5] for pv in param_vals]).T
        offsets = 2**arange(5)
        ft.plot0(fr[fr>5], A*(offsets[np.newaxis,:]),
                 ax=ax, col_cyc=cycler(color=param_colors),
                 xlabel="Frequency (Hz)", ylabel = "Amplitude", xlim=(10,300),
                 plot_fun = "loglog", title="Modes" if ip == 0 else "")

        # Ploting the amplitude spectra
        logger.info(f"Annotating amplitude spectra peaks  for '{param}'.")
        for iv, val in enumerate(param_vals):
            pf = sweep.peak_freqs[param, val, fld][:2] # We only want the first two peaks
            pv = sweep.peak_vals[param, val, fld][:2]
            ax.plot(pf, pv*offsets[iv], "rv", markersize=4, linewidth=1)

        logger.info(f"Plotting decay time courses '{param}'.")                
        ax = plt.subplot(gs[row:row+2,4])
        D = np.array([sweep.mean_rmse[param, pv, fld_rmse] for pv in param_vals]).T
        ft.plot0(sweep.t_decay, D @ diag([10**(i/2) for i in range(5)]),
                 ax=ax, col_cyc=cycler(color=param_colors),
                 xlabel="Time / sec.", ylabel = "RMSE",
                 plot_fun = "semilogy", title="RMSE Timecourse", plot_args={"linewidth":1})        

        logger.info(f"Plotting decay time constants for '{param}'.")
        ax = plt.subplot(gs[row:row+2,5])
        decays = np.array([sweep.decay_tau[param, pv, fld_rmse] for pv in param_vals])
        ft.plot0(log10(param_vals), decays, 
                 ax=ax, col_cyc=cycler(color=[[0,0,0,0.25]]),
                 xlabel=r"{}".format(ttls[param]), ylabel = r"$\tau$ (sec.)",
                 plot_fun = "plot", title="Decay Time Constant" if ip == 0 else "", plot_args={"linestyle":":", "linewidth":1})
        plt.scatter(log10(param_vals), decays, s=15, c = param_colors)
        ax.set_xticks(log10(param_vals))
        ax.set_xticklabels([f"{pv:1.3g}" for pv in param_vals], rotation=45)

            
    plt.tight_layout(pad = 0, w_pad = -0.5, h_pad=0.5)

    fig_file = f"SI_effect_of_params_on_{pop}s.pdf"
    logger.info(f"Saving as {fig_file}.")    
    plt.savefig(fig_file, bbox_inches="tight")
    
    logger.info("{} finished.".format(inspect.stack()[0][3]))    

def plot_effect_on_oscillations(sweep, pop, which_params, ton = 0.1, seed = 1, which_trial_to_plot = 0, hue_spread = 0.1):
    logger.info("{} started.".format(inspect.stack()[0][3]))
    
    pop = pop.lower()
    if pop not in ["mc", "gc", "pg"]:
        raise ValueError(f"'pop' must be one of 'mc', 'gc', or 'pg', was 'f{pop}'.")
    
    random.seed(seed)


    pop_tr = {"mc":"La", "pg":"Mu", "gc":"X"} # Translate population to data fields
    fld    = pop_tr[pop]
    fld_rmse = fld.lower() + "_rmse"
    ttls_esc  = {"tau_mc": "\\tau_\\lambda", "tau_pg": "\\tau_\\mu", "tau_gc":"\\tau_v",
                 "sd":"\\sigma", "ga":"\\gamma", "be":"\\beta"}    
    ttls_tex  = {p:f"${v}$" for p,v in ttls_esc.items()}

    t  = sweep.get_time()
    fs = int(round(1/(t[1] - t[0])))
    logger.info(f"fs: {fs}")

    ix = sweep.ix
    fr = sweep.fr

    loaded_params = sweep.get_loaded_params()
    if any([p not in loaded_params for p in which_params]):
        raise ValueError(f"Some desired parameters {which_params} not in loaded params {loaded_params}.")
    n_params      = len(sweep.get_param_vals(which_params[0]))

    cols_list, cols_dict = ft.Scols(pop)
    # Color for the central parameter values
    center_color = cols_dict[12] #cols_dict[sweep.S]    
    # Colors for other example traces in the background
    gray_colors  = [ft.set_alpha(0.75)(ft.cm.gray(rand()*0.25+0.75)) for i in range(100)]
    param_colors = [ft.set_alpha(0.75)(ft.color_for_param(center_color, ip, start = 1/4, end =1)) for ip in np.linspace(-1,1, n_params)]
    param_colors = [ft.set_alpha(0.75)(cols_dict[S]) for S in [5,10,15,20,25]]
    #param_colors = ft.gen_hue_spread_colors(center_color, n_params, hue_spread = hue_spread, alpha = 0.75)

    # Limits and ticks for the plots showing how 
    # the amplitude spectrum peaks vary with each parameter
    freq_fits_ylims  = {"mc":[log(10), log(175)],
                        "pg":[log(10), log(175)],
                        "gc":[log(10), log(250)]}

    freq_fits_yticks = {"mc":[10,20,50,100],
                        "pg":[10,20,50,100],
                        "gc":[10, 20,50,100,200]}

    amp_spec_ylims = {"mc":[0.1,500],
                      "pg":[0.5,1000],
                      "gc":[0.005,100]}

    # Offset the amplitude spectra for clarity    
    offsets = {"mc":2.**arange(5),
               "pg":2.**arange(5),
               "gc":2.**arange(-5,5,2)}

    # Xlimits for the time course plots
    xlims = {"mc":[-0.01, 0.2],
             "pg":[-0.01, 0.2],
             "gc":[-0.01, 0.2]}


    fig = plt.figure(figsize=(8,6))
    ft.apply_styles()
    
    gs = GridSpec(3,5)

    ax = []            
    for ip, param in enumerate(which_params):
        ttl = ttls_tex[param]
        param_vals = sorted(sweep.get_param_vals(param))
        ymin, ymax = [inf, -inf]
        row = 2*ip
        for iv, pv in enumerate(param_vals[::2]):
            ax.append(plt.subplot(gs[ip,iv]))
            X = copy(sweep[param, pv, fld][which_trial_to_plot]);
            X = X[:, ix] if pop == "gc" else X[:,0,:] # Plot the active GCs, or the sisters from the first glomerulus
            nx = X.shape[1]
            colors_list = gray_colors[:nx-1] + [param_colors[param_vals.index(pv)]]
            col_cyc = cycler(color=colors_list)
            #ft.plot0(t - ton, X, ax=ax[-1], col_cyc = col_cyc, xlabel =  "Time (sec.)", title=f"{ttls_tex[param]}: {pv:g}",
            #         xlim=(-0.01,0.2) if pop != "gc" else (-0.01,0.2))
            ttl = f"{ttls_tex[param]} = {pv:g}" 
            ft.plot0(t - ton, X[:,-1], ax=ax[-1], col_cyc = colors_list[-1],
                     xlabel =  "Time (sec.)", ylabel = f"{pop.upper()} activity" if iv == 0 else None,
                     xlim = xlims[pop],
                     plot_args = {"label":ttl})
            ymin = min(ymin, plt.ylim()[0])
            ymax = max(ymax, plt.ylim()[1])
            ax[-1].legend(frameon=False, fontsize=8, loc="lower right", borderpad=0, borderaxespad=0.25, handlelength=0)#

        # Ploting the amplitude spectra
        logger.info(f"Plotting amplitude spectra for '{param}'.")
        ax.append(plt.subplot(gs[ip,3]))
        A = np.array([sweep.amplitude_spectrum[param, pv, fld][fr>5] for pv in param_vals]).T
        ind_10hz = argmin(np.abs(fr[fr>5] - 10))
        scale = copy(A[ind_10hz])
        A /= scale

        ft.plot0(fr[fr>5], A*(offsets[pop][np.newaxis,:]),
                 ax=ax[-1], col_cyc=cycler(color=param_colors),
                 xlabel="Frequency (Hz)", ylabel = "Scaled Amp.", xlim=(10,300), ylim=amp_spec_ylims[pop],
                 ylabel_args = {"labelpad":-1},
                 plot_fun = "loglog")

        # Annotating with peaks
        logger.info(f"Annotating amplitude spectra peaks  for '{param}'.")
        for iv, val in enumerate(param_vals):
            for ipk, marker in enumerate(["v","^"]):
                pf = sweep.peak_freqs[param, val, fld][ipk] # We only want the first two peaks
                pv = sweep.peak_vals[param, val, fld][ipk] / scale[iv]
                ax[-1].plot(pf, pv*offsets[pop][iv], "r" + marker, markersize=4, linewidth=1)

        logger.info(f"Computing fits of low and high frequency peaks to '{param}'.")            
        ax.append(plt.subplot(gs[ip,4]))
        xx     = log(array(param_vals))
        yy_low = array([log(sweep.peak_freqs[param, pv, fld][0]) for pv in param_vals]) # low peaks
        yy_high= array([log(sweep.peak_freqs[param, pv, fld][1]) for pv in param_vals]) # high peaks
        p_low  = polyfit(xx, yy_low,  1) # linear fits
        p_high = polyfit(xx, yy_high, 1)

        fit_high = p_high[0]*xx + p_high[1]
        label0 = "${}".format(ttls_esc[param]) + "^{"        
        label = label0 + "{0:0.2g}".format(p_high[0]) + "}$"    
        ft.plot0(xx,fit_high, col_cyc="lightgray", plot_args={"label":label})

        fit_low = p_low[0]*xx + p_low[1]
        label = label0 + "{0:0.2g}".format(p_low[0]) + "}$"
        ft.plot0(xx, fit_low, col_cyc="lightgray", plot_args={"linestyle":"--", "label":label})

        ft.plot0(xx, yy_low,  ax = ax[-1], mode = "scatter",                 
                 ylabel="Freq. (Hz)", ylabel_args={"labelpad":-1},
                 plot_args={"marker":"v","markersize":4}, col_cyc = cycler(color=param_colors))
        
        ft.plot0(xx, yy_high, ax = ax[-1], mode = "scatter",
                 plot_args={"marker":"^","markersize":4}, col_cyc = cycler(color=param_colors),
                 xlabel=ttls_tex[param])

        plt.ylim(freq_fits_ylims[pop])
        yt = freq_fits_yticks[pop]
        ax[-1].set_yticks([log(y) for y in yt])
        ax[-1].set_yticklabels([str(y) for y in yt])
        ax[-1].set_xticks([log(param_vals[i]) for i in [0,2,-1]])
        ax[-1].set_xticklabels(["{0:g}".format(param_vals[i]) for i in [0,2,-1]])
    
        ax[-1].legend(frameon=False, labelspacing=-0.25, fontsize=8, loc="lower left" if p_high[0]<0 else "upper left", borderpad=0, borderaxespad=0.5)#
                
    plt.tight_layout()
    for i in range(3):    
        renderer = fig.canvas.get_renderer()
        ft.tighten_row(ax[i*5:(i+1)*5], fig=fig, renderer = renderer, h_w = 0.0)
    
    ft.label_axes([ax[i] for i in [0,3,4,5,8,9,10,13,14]], "ABCDEFGHI",fontsize=14, verticalalignment="center", horizontalalignment="left",fontweight="bold")

    fig_file = f"{pop.upper()}_oscillations.pdf"
    logger.info(f"Saving as {fig_file}.")    
    plt.savefig(fig_file, bbox_inches="tight")
    
    logger.info("{} finished.".format(inspect.stack()[0][3]))    

    
    
