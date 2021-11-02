# This script contains the fucntions for loading the data for and plotting the effects of leaky PGs
import os
import logging
import numpy as np
from numpy import *

from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.gridspec import GridSpec
from matplotlib import cycler
from matplotlib import ticker
from tqdm import tqdm
import inspect

import util
from olfactory_bulb import get_x_true, OlfactoryBulb
import datatools as dt
import figtools as ft
import figfuns as ff
import pdb

logger = util.create_logger("effect_of_leaky_pgs")
INFO   = logger.info

plt.style.use("default")
plt.style.use("./ms.mplstyle")

rand = random.rand

def compute_gap(x, ind_active, normalize=True, reg=1e-12):
    # X should be in cells x time format
    ind_inactive = list(set(range(len(x))) - set(ind_active))
    x_in = np.min(x[ind_active], axis=0)
    x_out= np.max(x[ind_inactive], axis=0)
    return (x_in - x_out)/(np.std(x,axis=0) + reg) if normalize else (x_in - x_out)

def compute_rmse_final():
    INFO(f"Computing final RMSE data.")        
    INFO(f"Loading data for random odours sweep.")
    root_folder = dt.data_folder("sweep_random_odours")        
    sweep = dt.FreeSweepDataset(root_folder, params = ["S","leak_pg"], load_on_demand = False)
    odours_data = sweep.load_sweep(warn_on_missing=False, vars=["x_MAP","x_final"])
    rmse_fun = lambda X, final: np.sqrt(np.mean((X["x_final"]*final - X["x_MAP"])**2))
    rmse_final = {(S,leak):np.array([rmse_fun(Xi,1) for Xi in XX]) for (S,leak), XX in odours_data.items()}
    rmse_orig  = {(S,leak):np.array([rmse_fun(Xi,0) for Xi in XX]) for (S,leak), XX in odours_data.items()}
    return rmse_orig, rmse_final
    
def compute_sister_correlations_data():
    INFO(f"Computing sister cell correlations data.")
    _collect_odour_responses = lambda data: array([datai["la_final"] for datai in data])

    def _compute_sister_correlations(la_final):
        n_odours, n_glom, S = la_final.shape
        ind_off_diag = np.where(np.triu(ones(S),1).flatten())[0]
        rho = [corrcoef(la_final[:,g,:].T).flatten()[ind_off_diag] for g in range(n_glom)]
        return array(rho).flatten()

    INFO(f"Loading data for random odours sweep.")
    root_folder = dt.data_folder("sweep_random_odours")        
    sweep = dt.FreeSweepDataset(root_folder, params = ["S","leak_pg", "seed"], load_on_demand = False)
    odours_data = sweep.load_sweep(warn_on_missing=False, vars=["x_MAP","la_final"])

    S_vals, leak_vals, seed_vals = map(unique, zip(*odours_data.keys()))
    INFO(f"Found data for {len(S_vals)}    S values, from {min(S_vals)} - {max(S_vals)}.")
    INFO(f"Found data for {len(leak_vals)} leak values, from {min(leak_vals)} - {max(leak_vals)}.")
    INFO(f"Found data for {len(seed_vals)} seed values, from {min(seed_vals)} - {max(seed_vals)}.")    

    INFO(f"Extracting la_final values for each S and leak combination.")
    la_finals = {(S,leak):[_collect_odour_responses(odours_data[(S,leak,seed)]) for seed in seed_vals] for S in S_vals[1:] for leak in leak_vals}

    n_trials           = len(la_finals[S_vals[-1],leak_vals[-1]])
    la_finals_instance = la_finals[S_vals[-1],leak_vals[-1]][0]
    INFO(f"Example la_finals shape for S={S_vals[-1]}, leak={leak_vals[-1]}:")
    INFO(f"  {n_trials} trials, size of the first: {la_finals_instance.shape}.")
    
    INFO(f"Computing all sister cell correlations.")    
    rho       = {(S,leak):array([_compute_sister_correlations(la_final) for la_final in la_finals[S,leak]]) for S in S_vals[1:] for leak in leak_vals}

    INFO(f"Averaging correlations across glomeruli .")        
    mean_rho  = {(S,leak):mean(rho[S,leak]) for S in S_vals[1:] for leak in leak_vals}

    INFO(f"Done computing sister cell correlations data.")
    return {"odours_data":odours_data,
            "S_vals":S_vals, "leak_vals":leak_vals, "seed_vals":seed_vals,
            "la_finals":la_finals, "correlations":rho, "mean_correlations":mean_rho}

def compute_sister_ratios_data(which_S = 25):
    INFO(f"Computing sister cell ratios data.")
    _collect_odour_responses = lambda data: array([datai["la_final"] for datai in data])

    INFO(f"Loading data for random odours sweep.")
    root_folder = dt.data_folder("sweep_random_odours")        
    sweep = dt.FreeSweepDataset(root_folder, params = ["S","leak_pg", "seed"], load_on_demand = False)
    which_param_combs = [p for p in sweep.param_value_combs_available if (p[sweep.pind["S"]] == which_S)]
    odours_data = sweep.load_sweep(param_combs = which_param_combs, warn_on_missing=False, vars=["x_MAP","la_final"])

    S_vals, leak_vals, seed_vals = map(unique, zip(*odours_data.keys()))
    INFO(f"Found data for {len(S_vals)}    S values, from {min(S_vals)} - {max(S_vals)}.")
    INFO(f"Found data for {len(leak_vals)} leak values, from {min(leak_vals)} - {max(leak_vals)}.")
    INFO(f"Found data for {len(seed_vals)} seed values, from {min(seed_vals)} - {max(seed_vals)}.")    

    INFO(f"Extracting la_final values for each S and leak combination.")
    la_finals = {(S,leak):[_collect_odour_responses(odours_data[(S,leak,seed)]) for seed in seed_vals] for S in S_vals[S_vals>1] for leak in leak_vals}
    n_trials           = len(la_finals[S_vals[-1],leak_vals[-1]])
    la_finals_instance = la_finals[S_vals[-1],leak_vals[-1]][0]
    INFO(f"Example la_finals shape for S={S_vals[-1]}, leak={leak_vals[-1]}:")
    INFO(f"  {n_trials} trials, size of the first: {la_finals_instance.shape}.")
    
    INFO(f"Computing all sister cell ratios.")    
    n_odours, n_glom, n_mc = la_finals_instance.shape
    ratio_fun = lambda mc: [mc[i]/(mc[j]+1e-12) for i in range(n_mc-1) for j in range(i+1,n_mc)]
    ratios = {ee:array([ratio_fun(la_finals[S_vals[-1],ee][tr][od,gl,:])
                        for od in range(n_odours)
                        for gl in range(n_glom)
                        for tr in range(n_trials)])
              for ee in leak_vals}
    INFO(f"Computing coefficients of variation.")            
    cvs    = {ee:np.std(rr,axis=1)/(np.abs(np.mean(rr,axis=1))+1e-12) for ee,rr in ratios.items()}

    INFO(f"Done computing sister cell ratios data.")
    return {"odours_data":odours_data,
            "S_vals":S_vals, "leak_vals":leak_vals, "seed_vals":seed_vals,
            "la_finals":la_finals, "ratios":ratios, "cvs":cvs}

    
def compute_x_MAP_data(leak_vals, max_S = 25):
    INFO(f"Computing x_MAP data for {leak_vals=}.")

    df, _ = dt.load_params_from_folder(dt.data_folder("sweep_qeps"))
    INFO(f"Loaded {len(df)} parameters.")
    
    def _get_params(S, leak):
        df_sub = df[(df.S==S) & (np.abs(df.leak_pg - leak)<1e-8)].sort_values("seed")
        files  = df_sub["file"].tolist()
        params = [dt.load_params_from_file(dt.data_folder("sweep_qeps"), f) for f in files]
        return params, files
    
    def _get_x_final(S, leak):
        return [np.load(os.path.join(dt.data_folder("sweep_qeps"), f[:-5], "x_final.npy")) for f in _get_params(S, leak)[1]]
    
    def _get_x_MAP(S, leak):
        return [np.load(os.path.join(dt.data_folder("sweep_qeps"), f[:-5], "x_MAP.npy"))   for f in _get_params(S, leak)[1]]
    
    def _get_x_MAP_leak(S, leak):
        params, files = _get_params(S, leak)
        x_MAP_leak = []
        with dt.SilenceLoggers("olfactory_bulb"):
            for i in range(len(params)):
                params[i]["ga"] += leak
                ob = OlfactoryBulb(**params[i])
                y  = ob.A @ get_x_true(params[i]["N"], params[i]["k"], spread=params[i]["spread"])
                x_MAP_leak.append(ob.run_exact(y, eps=5e-13)["x"])
        return x_MAP_leak

    leak_pg = sorted(list(set(df["leak_pg"])))
    INFO(f"Found {len(leak_pg)} leak_pg values from {leak_pg[0]} - {leak_pg[-1]}.")
    
    S_vals  = sorted(list(set(df["S"])))
    S_vals  = [S for S in S_vals if S<= max_S]
    INFO(f"Found {len(S_vals)} S values from {S_vals[0]} - {S_vals[-1]}.")

    INFO(f"Loading x_MAP.")
    # Load the x_MAP for the leak = 0 case.
    # We've compute the MAP values for leak != 0 as well.
    # We've just ignored the leak value in those cases.
    # All the x_MAP values should be the same, because it shouldn't depend on S or leak,
    # only on the trial, because the A matrices will be different...
    x_MAP   = _get_x_MAP(S_vals[0], 0)

    # ...so now make sure that the values are in fact all the same
    INFO(f"Checking that x_MAP values are independent of leak.")                
    for S in S_vals:
        for leak in leak_vals:
            x_MAP_this = _get_x_MAP(S, leak)
            if len(x_MAP_this) == 0:
                raise KeyError(f"No x_MAP data found for ({S=},{leak=}).")
            assert(allclose(x_MAP, x_MAP_this))
    INFO(f"x_MAP values were all the same.")

    INFO(f"Loading x_final.")    
    x_final = {(S, leak):_get_x_final(S, leak) for S in S_vals for leak in leak_vals}

    INFO(f"Computing x_final_gap")
    ind_nz          = where(x_MAP[0]>0.5)[0]
    x_final_gap     = {k:compute_gap(array(xf).T,ind_nz)                  for k,xf in x_final.items()}
    x_final_gap_abs = {k:compute_gap(array(xf).T,ind_nz, normalize=False) for k,xf in x_final.items()}

    INFO(f"Computing x_final amplitude and norms.")
    x_final_amp = {k:np.max(array(xf),axis=1) for k,xf in x_final.items()}
    x_final_l1  = {k:np.sum(abs(array(xf)),axis=1) for k,xf in x_final.items()}
    x_final_l0  = {k:np.sum(abs(array(xf))>1e-2,axis=1) for k,xf in x_final.items()}        
    
    INFO(f"Computing x_MAP_leak.")        
    x_MAP_leak = {}
    configs = [(S, leak) for S in S_vals for leak in leak_vals]
    for (S, leak) in tqdm(configs):
        x_MAP_leak[S,leak] = _get_x_MAP_leak(S,leak)

    INFO(f"Computing errors relative to the MAP solution.")
    err_x_MAP      = {leak:[mean((array(x_final[S,leak]) - array(x_MAP))**2,axis=1) for S in sorted(S_vals)] for leak in leak_vals}

    INFO(f"Computing errors relative to the MAP solution with γ → γ + ε.")    
    err_x_MAP_leak = {leak:[mean((array(x_final[S,leak]) - array(x_MAP_leak[S,leak]))**2,axis=1) for S in sorted(S_vals)] for leak in leak_vals}

    INFO(f"Done computing x_MAP_data.")
    return {"x_MAP":x_MAP,
            "x_final":x_final,         "x_final_gap":x_final_gap,       "x_final_gap_abs":x_final_gap_abs,
            "x_final_amp":x_final_amp, "x_final_l0":x_final_l0, "x_final_gap_l1":x_final_l1,            
            "x_MAP_leak":x_MAP_leak,
            "err_x_MAP":err_x_MAP, "err_x_MAP_leak":err_x_MAP_leak,
            "S":S_vals}
    
leak_vals = list(np.round(list(arange(0,1,0.1))+[1,1.5,2],2))

def load_data(S_leak = [(8,leak) for leak in leak_vals],
              S_leak_final = [(S,leak) for S in [1,2,8,25] for leak in leak_vals],
              x_MAP_leak_vals = leak_vals, which_seed = None):
    INFO("Started {}.".format(inspect.stack()[0][3]))
    
    INFO(f"Loading data for {S_leak=} and seed={which_seed}.")    
    results = {(S, leak_pg):ff.load_data(S = S, leak_pg = leak_pg, which_seed = "all") for (S,leak_pg) in S_leak}
        
    INFO(f"Updating RMSE values for results with leaky pgs against leak = 0 values.")
    make2d = lambda X: X.reshape(X.shape[0], -1)
    
    # For each leak_pg, we have the results for each seed.
    S_vals = list(set([p[0] for p in S_leak]))
    for S in S_vals:
        if (S,0) not in results:
            raise KeyError(f"No results loaded for leak=0, {S=}. This is required.")
    results_no_leak = {S:results[S,0] for S in S_vals}
    seed_vals       = list(results[S_vals[0], 0].keys())
    default_params  = dt.load_default_params()
    
    for (S, leak), results_for_leak in results.items():
        if leak==0:
            continue
        
        this_seed_vals = list(results_for_leak.keys())
        if set(seed_vals) != set(this_seed_vals):
            raise ValueError(f"Seed values for ({S=}, {leak=}): {this_seed_vals} don't match those for ({S=} leak=0): {seed_vals}")
        INFO(f"Seed values for {S=:<3d}, {leak=:<1.1f} match those for leak=0.")
        for seed, result_per_seed in results_for_leak.items():
            # Compute RMSEs
            U = result_per_seed["X"]
            u_target = results_no_leak[S][seed]["x_MAP"].flatten()             
            results[S, leak][seed]["rmse"]  = np.sqrt(np.mean((make2d(U) - u_target)**2,axis=1))

            # Now compute the x_MAP values for gamma + eps
            params = dict(default_params)
            params.update(dt.load_params_from_file(dt.data_folder("sweep_conc_spread"), results[S,leak][seed]["file"]))
            params["ga"] += leak
            results[S, leak][seed]["x_MAP_leak"] = dt.get_x_MAP_for_params(params, silent_ob = True, eps=5e-13)
                        
    INFO(f"Collect RMSE values for each leak value.")
    rmse = {(S, leak):array([res[seed]["rmse"] for seed in seed_vals]) for (S, leak),res in results.items()}

    INFO(f"Computing gap.")    
    x_true = get_x_true(1200, 3, spread=0.4)
    ind_x_true = where(x_true)[0]
    gap = {(S, leak):array([compute_gap(res[seed]["X"].T, ind_x_true, reg=1e-12) for seed in seed_vals]) for (S, leak), res in results.items()}

    # Load the x_finals
    x_finals = {(S,leak_pg): array([Xi["x_final"] for tr, Xi in ff.load_data(S = S, leak_pg = leak_pg, which_seed = "all").items()]) for (S,leak_pg) in S_leak_final}
    # Compute the gap values at the last iteration
    gap_final = {k:compute_gap(Xi.T, ind_x_true, reg=1e-12) for k, Xi in x_finals.items()}
    
    # The above values were computed for the demo cases,
    # with x_true having a non-zero spread. This is from the
    # sweep_conc_spread runs.
    #
    # Now compute the results for all the sweep_qeps runs, where we
    # tried a variety of S values, and a few eps values, and
    # there was no spread.
    INFO(f"Computing x_MAP_data for {x_MAP_leak_vals=}.")        
    x_MAP_data = compute_x_MAP_data(x_MAP_leak_vals) # Uses sweep_qeps

    # Load the data for the runs where we tried different odours.
    # This was to see how the sister cells relate at convergence in the leaky setting.
    sister_ratios_data  = compute_sister_ratios_data() # Uses sweep_random_odours

    # Compute the rmse_[orig|final] data
    rmse_orig, rmse_final = compute_rmse_final() # Uses sweep_random_odours
    
    INFO("Done {}.".format(inspect.stack()[0][3]))        
    return {"results":results, "rmse":rmse, "gap":gap, "gap_final":gap_final,
            "x_MAP_data":x_MAP_data, "sister_ratios_data":sister_ratios_data,
            "rmse_orig":rmse_orig, "rmse_final":rmse_final, "x_finals":x_finals}

def plot_effect_of_leaky_pgs(data, iglom = 2, cm_mc = ft.pop_cmaps["mc"], cm_gc=ft.pop_cmaps["gc"], S_plot = 8, t_on = 0.1, subplots = None, label_order = None, fig_name = "effect_of_leaky_pgs"):
    INFO("Started {}.".format(inspect.stack()[0][3]))

    xl = (-0.01,0.3)    
    xt = arange(0,0.51,0.1)

    yl = [[-2,0.1],[-7.2,4.1]]
    yt = [arange(-5, 2, 1), arange(-6,4.1,2)]    

    mc_cols = [cm_mc(val) for val in [1, 0.5]]

    _, gc_S_cols = ft.Scols("gc")
    
    S_leak = sorted(data["results"])
        
    seed_vals = sorted(data["results"][S_leak[0]])

    pos_leak = {0:1, 0.2:0.85, 0.5:0.6, 1:0.4, 1.5:0.2, 2:0.1}
    gc_leak_cols = {leak:cm_gc(pos) for leak, pos in pos_leak.items()}

    leak_vals = array(sorted(list(set([leak for (S,leak) in data["results"].keys() if S == S_plot]))))
    
    if subplots is None:
        plt.figure(figsize=(8,10))

    gs = GridSpec(5,4)
    
    ax = []
    ret_ax = {}
    
    if subplots is None:
        INFO(f"Plotting sister cell time courses for S = {S_plot}.")
        for i, leak_val in enumerate(leak_vals[leak_vals>0]):
            INFO(f"  Plotting for leak = {leak_val}.")        
            results = data["results"][S_plot,leak_val][seed_vals[0]]
            
            # Top panel 4 sisters 
            random.seed(0)
            n_mc   = results["La"].shape[-1]
            
            plot_args = {"linewidth":2}

            gs_sub = gs[i,2:]
            axi, h = ft.plot0(results["T"] - t_on, 0*results["T"], ax=plt.subplot(gs_sub), col_cyc="lightgray", plot_args={"linewidth":1})    
            axi, h = ft.plot0(results["T"] - t_on, results["La"][:,iglom,:],
                             ax = axi, col_cyc = cycler(color=[ft.set_alpha(0.5)(mc_cols[i])]),
                             xticks = xt, yticks = yt[i],
                             xlim   = xl, ylim   = yl[i],
                             xlabel = "Time / sec.", ylabel = "MC activity",
                             plot_args = plot_args)
    
            ax.append(axi)
            ret_ax[leak_val] = ax[-1]
    
    get_gridspec = lambda name, default: default if subplots is None else subplots[name]
    do_plot      = lambda names: subplots is None or any([name in subplots for name in names])
    legend = lambda *args, **kwargs: plt.legend(*args, frameon=False, labelspacing=0, fontsize=8, borderpad=0, **kwargs)
    
    if do_plot([(4, 1), (4,10), (80, 10)]):
        INFO(f"Plotting granule cell time courses.")        
        gc_params = {(4, 1):gs[0,:2], (4,10):gs[1,:2], (80,10):gs[4,:2]}
        for (S, leak_val), gsi in gc_params.items():
            if subplots is not None:
                if (S, leak_val) not in subplots:
                    continue
                else:
                    # Override the gridpsec with the one provided
                    # when specific subplots have been specified.
                    gsi = subplots[(S, leak_val)]
                    
            INFO(f"  Plotting for S = {S:<2d} leak={leak_val:>d}.")        
            results = data["results"][S,leak_val][seed_vals[0]]
            gc_cols = ft.gen_gc_cols(central_color = gc_leak_cols[leak_val], hue_spread = 0, alpha_active=0.75)
            axi, h  = ft.plot0(results["T"] - t_on, results["X"],
                               ax = plt.subplot(gsi), col_cyc = cycler(color=gc_cols),
                               xlim = xl, ylim = (-0.1, 2),
                               xlabel="Time / sec.",
                               xticks = xt, yticks = [0,1,2],
                               yticklabels=["0","1","2"],ylabel  = "GC activity")
            ax.append(axi)
            ret_ax[(S, leak_val)] = ax[-1]
            x_MAP  = results["x_MAP"]        
            [plt.plot(plt.xlim()[1], x, "r<", linewidth=1) for x in x_MAP[x_MAP>0.5]]
            if (S, leak_val) == (80,10):
                x_MAP_leak = results["x_MAP_leak"]
                [plt.plot(plt.xlim()[1], x, "r<", linewidth=1, markerfacecolor="none") for x in sorted(x_MAP_leak)[-3:]]            
                            
                
    if do_plot(["rmse"]):
        INFO(f"Plotting RMSE relative to the x_MAP value for each leak value.")
        rmse = data["rmse"]
        leak_vals = sorted([leak for (S,leak) in rmse.keys() if S == S_plot])
        ax_rmse = plt.subplot(get_gridspec("rmse", gs[2,:2]))
    
        xl = (-t_on,0.5)
        t = data["results"][S_plot, 0][0]["T"]
        
        h = []
        for i, leak in enumerate(reversed(leak_vals)):
            rmsei = rmse[S_plot, leak]/rmse[S_plot, leak][:,0][:,np.newaxis]
            offset = 1 #(10**-i)
            yy = offset * rmsei
            my = mean(yy, axis=0)
            sy = std(yy,axis=0)
            color = gc_leak_cols[leak]
            col_cyc = cycler(color=[ft.set_alpha(0.25)(color)])
            ft.plot0(t - t_on, yy.T, ax=ax_rmse, col_cyc=col_cyc, plot_fun="semilogy", xlim=xl, ylim=(1e-8,2), xlabel="Time (sec.)", plot_args={"linewidth":1})                
            _, hi = ft.plot0(t - t_on, my, ax=ax_rmse, col_cyc=cycler(color=[color]), plot_fun="semilogy", xlim=xl, ylim=(1e-3,2), plot_args={"label":f"$\\varepsilon$={leak}"})            
            h.append(hi)
            INFO(f"  Plotted {len(yy)} traces for {leak=}.")

        legend(loc="lower left")
        #ax_rmse.yaxis.set_ticks([10**-i for i in range(0,12,2)])
        ax_rmse.set_xlabel("Time (sec.)", labelpad=-5)        
        ax_rmse.set_xlim(-0.01,0.3)
        ax_rmse.set_xticks(arange(0,0.31,0.1))
        ax_rmse.set_ylabel("RMS error", labelpad=-2)
        ret_ax["rmse"] = ax_rmse

    if do_plot(["gap"]):        
        INFO(f"Plotting gap for each leak value.")
        gap = data["gap"]
        leak_vals = [0,1.0,2.0]
        ax_gap = plt.subplot(get_gridspec("gap",gs[3,:2]))
    
        xl = (-t_on,0.5)
        t = data["results"][S_plot, 0][0]["T"]    
        h = []
        plt.plot(t - t_on, 0*t, ":", color="lightgray", linewidth=1) # Plot a horizontal at 0

        for i, leak in enumerate(reversed(leak_vals)):
            t = data["results"][S_plot, leak][0]["T"]                
            gapi = gap[S_plot, leak]
            yy = gapi
            my = mean(yy, axis=0)
            sy = std(yy,axis=0)
            color = gc_leak_cols[leak]
            col_cyc = cycler(color=[ft.set_alpha(0.25)(color)])
            ft.plot0(t - t_on, yy.T, ax=ax_gap, col_cyc=col_cyc, xlim=xl, xlabel="Time (sec.)", plot_args={"linewidth":1})                
            _, hi = ft.plot0(t - t_on, my, ax=ax_gap, col_cyc=cycler(color=[color]), xlim=xl, xlabel="Time (sec.)", plot_args={"label":f"$\\varepsilon$={leak}"})
            h.append(hi)
            INFO(f"  Plotted {len(yy)} traces for {leak=}.")
    
        legend(loc="lower right")
        ax_gap.set_xlim(-0.01,0.3)
        ax_gap.set_xticks(arange(0,0.31,0.1))
        ax_gap.set_ylabel("Normalized gap", labelpad=-2)
        ret_ax["gap"] = ax_gap

    x_MAP_data = data["x_MAP_data"]    
    S_vals     = x_MAP_data["S"]
    leaks_to_plot = [0.2,0.5,1.0, 2.0]
    col_cyc = cycler(color=[gc_leak_cols[leak] for leak in leaks_to_plot])
    col_cyc_r = cycler(color=[gc_leak_cols[leak] for leak in reversed(leaks_to_plot)])
    labels = [f"$\\varepsilon$={leak}" for leak in reversed(leaks_to_plot)]
    x_vals = S_vals
        
    if do_plot(["amp", "card", "gap_z"]):
        INFO(f"Plotting x_final amplitude and density.")
        
        if do_plot(["amp"]):
            ax_amp = plt.subplot(get_gridspec("amp", gs[2,2]))
            y_vals  = array([[mean(x_MAP_data["x_final_amp"][S,leak]) for S in x_vals] for leak in reversed(leaks_to_plot)])
            
            ft.plot0(x_vals, y_vals.T, col_cyc=col_cyc_r, plot_fun="semilogx",
                     xlabel="S", ylabel="Amplitude", ylim=[0,1])
            legend(labels, loc="lower left")
            ret_ax["amp"] = ax_amp

        if do_plot(["card"]):
            ax_l0  = plt.subplot(get_gridspec("card", gs[2,3]))
            y_vals = array([[mean(x_MAP_data["x_final_l0"][S,leak]) for S in x_vals] for leak in reversed(leaks_to_plot)])
            ft.plot0(x_vals, y_vals.T, col_cyc=col_cyc_r, plot_fun="semilogx",
                     xlabel="S", ylabel="Cardinality")
            legend(labels, loc="upper left")
            ret_ax["card"] = ax_l0

        if do_plot(["gap_z"]):
            INFO(f"Plotting x_final normalized gap.")
            ax_gap_z = plt.subplot(get_gridspec("gap_z", gs[3,2]))
            mean_gaps = {(S,leak):mean(gap_final_k) for (S,leak), gap_final_k in data["gap_final"].items()}
            x_vals = sorted(unique([k[0] for k in mean_gaps.keys()]))
            y_vals = array([[mean_gaps[S,leak] for S in x_vals] for leak in reversed(leaks_to_plot)])
            ft.plot0(x_vals, y_vals.T, col_cyc=col_cyc_r, plot_fun="semilogx",
                     xlabel="S", ylabel="Normalized gap")
            legend(labels, loc="lower left")
            ax_gap_z.set_ylim(0, ax_gap_z.get_ylim()[-1])
            ax_gap_z.set_yticks(np.arange(0,21,5))
            ret_ax["gap_z"] = ax_gap_z
            
    if do_plot(["rel", "rel_approx"]):
        INFO(f"Plotting error relative to the leak=0 case.")
        
        mse_x_MAP = x_MAP_data["err_x_MAP"]
        mu_x_MAP = {leak:[mean(sqrt(erri)) for erri in errs] for leak, errs in mse_x_MAP.items()}
        sd_x_MAP = {leak:[ std(sqrt(erri)) for erri in errs] for leak, errs in mse_x_MAP.items()}

        if do_plot(["rel"]):
            ax_err1 = plt.subplot(get_gridspec("rel",gs[4,2]))
            x_vals  = S_vals[1:]
            y_vals  = array([mu_x_MAP[leak][1:]/mu_x_MAP[leak][1] for leak in reversed(leaks_to_plot)])        
            ft.plot0(x_vals, y_vals.T, ax=ax_err1, col_cyc=col_cyc_r, xlabel="S", ylabel="Relative RMS error",
                     plot_fun="semilogx", plot_args={"linewidth":2}, yticks=[1,2,3] )
            legend([f"$\\varepsilon$={leak}" for leak in reversed(leaks_to_plot)], loc="upper left")
            ret_ax["rel"] = ax_err1

        if do_plot(["rel_approx"]):
            INFO(f"Plotting error relative to the leak=0, γ→γ+ε  case.")    
            err_x_MAP_leak = x_MAP_data["err_x_MAP_leak"]
            INFO(f"  Averaging across {len(err_x_MAP_leak[10])} trials.")
            mu_x_MAP_leak = {leak:array([mean(sqrt(erri)) for erri in errs]) for leak, errs in err_x_MAP_leak.items()}
            sd_x_MAP_leak = {leak:array([std(sqrt(erri))  for erri in errs]) for leak, errs in err_x_MAP_leak.items()}    
        
            ax_err2 = plt.subplot(get_gridspec("rel_approx",gs[4,3]))
            x_vals  = S_vals[1:]
            y_vals  = array([mu_x_MAP_leak[leak][1:]/mu_x_MAP_leak[leak][1] for leak in reversed(leaks_to_plot)])
            ft.plot0(x_vals, y_vals.T, ax=ax_err2, col_cyc=col_cyc_r, xlabel="S", ylabel="Relative RMS error", plot_fun="semilogx", plot_args={"linewidth":2}, )
            legend([f"$\\varepsilon$={leak}" for leak in reversed(leaks_to_plot)], loc="upper right")
            ret_ax["rel_approx"] = ax_err2

    if do_plot(["ratios"]):
        INFO(f"Plotting sister cell ratios.")
        sister_ratios_data = data["sister_ratios_data"]
        S_vals    = sister_ratios_data["S_vals"]
        leak_vals = sister_ratios_data["leak_vals"]
        leak_vals = leak_vals[leak_vals<=2]
        cvs       = sister_ratios_data["cvs"]
        which_pcs = [5,25,50,75,95]
        pcs       = {ee:np.percentile(cvse, which_pcs) for ee,cvse in cvs.items()}
        x_vals    = sorted(leak_vals)
        ax_ratios = plt.subplot(get_gridspec("ratios", gs[3,3]))
        for S in S_vals:
            plt.semilogy([x_vals, x_vals], [[pcs[ee][1] for ee in x_vals], [pcs[ee][3] for ee in x_vals]], color = gc_S_cols[S], linewidth=1)
            plt.plot(x_vals, [pcs[ee][2] for ee in x_vals], "o", markersize=5, color = gc_S_cols[S], label = f"{S=}")
        plt.ylim(1e-3,1e0)
        plt.xlim(-0.1,2.1)
        ax_ratios.set_xticks(leak_vals)
        ax_ratios.set_xticklabels([f"{lv:0.1f}" if ((not mod(int(lv*10), 2)) or lv>=1) else "" for lv in leak_vals])        
        plt.xlabel("$\\varepsilon$")
        plt.ylabel("Coefficient of variation")
        
        if len(S_vals) > 1:
            legend(loc="upper left")
        ret_ax["ratios"] = ax_ratios

    # This isn't plotted if subplots is None        
    if "rmse_final" in subplots: 
        INFO(f"Plotting final rmse data.")
        rmse_final = data["rmse_final"]
        rmse_orig  = data["rmse_orig"]
        S_vals, leak_vals = zip(*list(rmse_final.keys()))
        S_vals    = np.unique(S_vals).astype(int)
        leak_vals = np.unique(leak_vals)
        leak_vals = leak_vals[leak_vals<=2]
        x_vals    = sorted(leak_vals[1:])
        y_vals    = array([[mean(rmse_final[S,leak]/rmse_orig[S,leak]) for leak in x_vals] for S in S_vals[1:]])
        ax_rmsef  = plt.subplot(subplots["rmse_final"])
        ft.plot0(x_vals, y_vals.T, ax=ax_rmsef, col_cyc=cycler(color=[gc_S_cols[S] for S in S_vals[1:]]),
                 plot_fun = "semilogy", xlabel="$\\varepsilon$", ylabel="Final RMS error",
                 xlim=(-0.1,2.1))
        ax_rmsef.set_xticks(leak_vals)
        ax_rmsef.set_xticklabels([f"{lv:0.1f}" if ((not mod(int(lv*10), 2)) or lv>=1) else "" for lv in leak_vals])
        legend([f"{S=}" for S in S_vals[1:]], loc="lower right")
        ret_ax["rmse_final"] = ax_rmsef

    if "x_final" in subplots:
        INFO(f"Plotting x_finals.")
        x_finals = data["x_finals"]
        ax_xf   = plt.subplot(subplots["x_final"])
        #mults = {0:1, 1:1, 10:2, 100:5}
        which_leaks = [0,0.5,1,2]
        mults = {l:1 for l in which_leaks}
        for i, leak in enumerate(which_leaks):
            ax_xf.plot(x_finals[25, leak][0]*mults[leak] - i*2, color=cm_gc(1-i/3.5))
        yt = [-5,-3,-1,1]
        ax_xf.set_yticks(yt)
        ax_xf.set_yticklabels(["Amplitude" if m == 1 else f"Amp. x {m}" for m in reversed(sorted(mults.values()))], rotation=90, fontsize=10)
        [ax_xf.text(1000, 0.75 - i*2, f"$\\varepsilon={leak}$") for i, leak in enumerate(which_leaks)]
        ax_xf.spines['left'].set_visible(False)
        ax_xf.tick_params(axis='y', length=0, pad=-1)
        ax_xf.set_xlim(0,1200)
        ax_xf.set_ylim(-6.1, 1.5)
        ax_xf.set_xlabel("Granule cell index")
        ret_ax["x_final"] = ax_xf

    INFO("Finalizing figure.")

    plt.tight_layout(pad=0, h_pad=1)
    
    label_axes_fun = lambda ax_list: ft.label_axes(ax_list, "ABCDEFGHIJKLMNOP",fontsize=14, verticalalignment="center", horizontalalignment="left",fontweight="bold") 

    if subplots is None:
        label_axes_fun([ax[2],ax[0],ax[3],ax[1]] + [ax_rmse, ax_amp, ax_l0] + [ax_gap, ax_gap_z, ax_ratios] + [ax[4],ax_err1, ax_err2])
    else:
        label_axes_fun([ret_ax[k] for k in (label_order if label_order is not None else sorted(subplots))])     
        
    fig_file = f"{fig_name}.pdf"
    INFO(f"Saving as {fig_file}.")    
    plt.savefig(fig_file, bbox_inches="tight")
    
    INFO("Finished {}.".format(inspect.stack()[0][3]))
    return ret_ax
    

        
    
    
