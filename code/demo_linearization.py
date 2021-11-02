import logging
import numpy as np
from numpy import *
from matplotlib import pyplot as plt
from matplotlib import cm

import util
import inspect
import datatools as dt
import figtools as ft
logger = util.create_logger("demo_linearization")
INFO   = logger.info

import test_linearization as TestLin


def load_data(Sn_vals = [("sparse", 8, 20), ("sparse", 25, 20), ("dense", 25, 55)], leak_vals = [0, 0.5, 1, 2, 5]):
    INFO("Started {}.".format(inspect.stack()[0][3]))
    data = {}
    with dt.SilenceLoggers("olfactory_bulb"):
        for leak in leak_vals:
            data[leak] = {}
            for i, (name, S, n) in enumerate(Sn_vals):
                INFO(f"Computing {leak=} ({name=}, {S=}, {n=}).")
                od = TestLin.SparseOdour() if name == "sparse" else TestLin.DenseOdour()
                od.name = name    
                TestLin._basic_setup(od, n = n, S=S, leak = leak, partitioned = False)
                ee = np.linalg.eigvals(od.H)
                pred_eigs, _ = TestLin.compute_predicted_eigenvalues(od)
                approx       = od.test_qxi_roots_approx(run_test = False)
                ob = od.ob
                nu_min = (1 - sqrt(n/ob.M))**2
                nu_max = (1 + sqrt(n/ob.M))**2
                nu_vals= linspace(nu_min, nu_max, 101)
                approx_for_nu = {nu:ob.approximate_sparse_odour_eigs(nu) for nu in nu_vals for nu in nu_vals}
                data[leak][(name, S, n)] = {"eigs":ee, "pred":pred_eigs, "approx":approx, "approx_for_nu":approx_for_nu}
        INFO("Finished {}.".format(inspect.stack()[0][3]))    
    return data
        
def plot(data, xl = [-25, 1], leak=0.5, plot_approx_for_nu=False,figsz=(8,6)):
    INFO("Started {}.".format(inspect.stack()[0][3]))
    mc_col = "dodgerblue" # ft.pop_cmaps["mc"](0.5)
    gc_col = "orange" # ft.pop_cmaps["gc"](0.6)

    styles = {"actual":                  {"marker":"x", "color":cm.gray(0.8),   "markersize":5, "linestyle":"None"},
              "qxi_low_roots_approx":    {"marker":"v", "color":gc_col,         "markersize":4, "linestyle":"None"},
              "qxi_high_roots_approx":   {"marker":"+", "color":gc_col,         "markersize":8, "linestyle":"None"},
              "q0xi_low_root_approx":    {"marker":"D", "color":"red",          "markersize":4, "linestyle":"None"},
              "q0xi_high_roots_approx":  {"marker":"^", "color":"red",          "markersize":4, "linestyle":"None"},
              "static":                  {"marker":"D", "color":gc_col,         "markersize":2.5, "linestyle":"None"},
              "v0_decaying":             {"marker":"D", "color":mc_col,         "markersize":2.5, "linestyle":"None"},
              "v0_oscillating":          {"marker":"+", "color":mc_col,         "markersize":8, "linestyle":"None"}
    }
    mew = 1.25
    for k, stylek in styles.items():
        if k!="actual":
            styles[k]["markeredgewidth"] = mew
    
    labels = {"qxi_low_roots_approx":"$\\xi^{low}$",
              "qxi_high_roots_approx":"$\\xi^{high}$",
              "q0xi_low_root_approx":"$\\xi^{low}_{\\nu=0}$",
              "q0xi_high_roots_approx":"$\\xi^{high}_{\\nu=0}$",
              "static": "decay$_\\mu$",
              "v0_decaying":"decay$_\\lambda$",
              "v0_oscillating":"$\\xi^{mp}$"}

    plt.figure(figsize=figsz)

    if leak not in data:
        raise KeyError(f"No data for {leak=}.")
    Sn_vals = [("sparse", 25,20), ("dense", 25,55)]
    if not all([Sn in data[leak] for Sn in Sn_vals]):
        raise KeyError(f"Data for some {Sn_vals=} not available.")
    ax = []
    for i, (name, S, n) in enumerate(Sn_vals):
        datai = data[leak][name, S, n]
        INFO(f"Plotting ({name=}, {S=}, {n=})")
        ee, pred, approx = [datai[fld] for fld in ["eigs", "pred", "approx"]]
        ax.append(plt.subplot(len(Sn_vals), 1, i+1))
        plt.plot([-25,1],[0,0], ":", linewidth=1, color="lightgray")
        plt.plot(real(ee), imag(ee), "x", **styles["actual"], label="actual" if i < 2 else None)                
        for fld in ["static", "v0_decaying", "v0_oscillating"]:
            if fld in pred:
                vals = sorted(pred[fld], key=imag)
                vals = [vals[0], vals[-1]]
                [plt.plot(real(ee), imag(ee), **styles[fld], label = labels[fld] if (not j and i < 2) else None) for j, ee in enumerate(vals)]
    
        nu_vals = sorted(datai["approx_for_nu"].keys())
        low     = array([datai["approx_for_nu"][nu]["low"] for nu in nu_vals])
        high    = array([datai["approx_for_nu"][nu]["high"] for nu in nu_vals])
        if name == "sparse":
            [[plt.plot(real(ee), imag(ee), **styles[fld], label=labels[fld] if (not j and i < 2) else None) for j, ee in enumerate(approx[fld])] for fld in ["qxi_low_roots_approx", "qxi_high_roots_approx"]]
        else:
            [[plt.plot(real(ee), imag(ee), **styles[fld], label=labels[fld] if (not j and (i < 2 or "q0" in fld)) else None) for j, ee in enumerate(approx[fld])] for fld in ["q0xi_low_root_approx", "q0xi_high_roots_approx","qxi_low_roots_approx", "qxi_high_roots_approx"]]

        if plot_approx_for_nu:
            plt.plot(real(low),  imag(low),  "k")
            plt.plot(real(high), imag(high), "k")        

        xt_ms = range(40,104,5)        
        plt.gca().set_xticks([-1000/xti for xti in xt_ms])
        plt.gca().set_xticklabels([str(xti) for xti in xt_ms])
        plt.xlim(xl)                    
        yt_Hz = np.arange(-400,500,200)
        plt.gca().set_yticks([yti*2*pi for yti in yt_Hz])
        plt.gca().set_yticklabels([str(yti) for yti in yt_Hz])
        
        #plt.ylim([-2000,2000])
        #plt.gca().set_yticklabels([])
        plt.legend(fontsize=9, frameon=False, labelspacing=0, borderpad=0, handletextpad=-0.5, loc="upper right")
        plt.xlabel("Time constant (msec.)", fontsize=12)        
        plt.ylabel("Frequency (Hz)", fontsize=12)
        plt.gca().tick_params(axis='both', which='major', labelsize=10)        
        #plt.gca().set_xticks([-15,-20])
        #yt = np.arange(-2000,2001,1000)
        #plt.gca().set_yticks(yt)
        #plt.gca().set_yticklabels([int(yti/1000) for yti in yt])
        #[plt.gca().spines["left"].set_visible(False), plt.gca().set_yticks([])]
    plt.tight_layout()
            
    ft.label_axes(ax, "AB", fontsize=14, fontweight="bold", dy=[0.0]*3 + [0]*3, dx = 0)
    file_name = f"demo_lin_{leak=:1.1f}" + ("_nuaprx" if plot_approx_for_nu else "") + ".pdf"
    plt.savefig(file_name, bbox_inches="tight")
    INFO(f"Wrote {file_name}.")
