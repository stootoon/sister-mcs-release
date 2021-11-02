import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm as colormaps
from matplotlib import colors as mcolors
from scipy.signal import find_peaks
from tqdm import tqdm
import logging

def create_logger(name, level = logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        for h in logger.handlers:
            logger.removeHandler(h)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(fmt='%(asctime)s %(module)24s %(levelname)8s: %(message)s', datefmt='%Y/%m/%d %H:%M:%S'))
    logger.addHandler(ch)
    return logger
    
def rms_error(x,y, verbose = True):
    err = np.sqrt(np.mean((x-y)**2))
    if verbose:
        print("RMS error: {}".format(err))

    return err
    
def spectrum(x, fs = 1., color = None, plot_fun = None, mean_subtract = False, mark_peak = False):
    if mean_subtract:
        x -= np.mean(x)
    f = fft(x)
    freqs = np.arange(len(x))/float(len(x))*fs
    if mark_peak:
        print("Peak AC frequency: {:.1f} Hz".format(freqs[(freqs>0)&(freqs<0.5*fs) ][np.argmax(abs(f[(freqs>0)&(freqs<0.5*fs)]))]))
    if plot_fun:
        plot_fun(freqs, abs(f), color=color) if color else plot_fun(freqs, abs(f))
        plt.xlim(0,0.5*fs)
        plt.xlabel("frequency / Hz")
        plt.ylabel("|X(f)|")
    return f, freqs

def plot_odor_response(results, which_x, which_la, which_mu, x_true = [], x_or_v = "x", plot_every = 1e-3, normalize=False, draw_mode = "tall"):
    t = results["t"]
    
    tplot = np.arange(t[0],t[-1], plot_every)
    iplot = np.array([np.argmin(np.abs(t-tp)) for tp in tplot])
    # PLOT EVERY
    
    X = results["x"][iplot, :]
    V = results["v"][iplot, :]
    La= results["la"][iplot,:,:]
    Mu= results["mu"][iplot,:,:]
    t = tplot
    
    nt,M,S = La.shape
    
    if type(which_x) is int:
        # Pick the GCs with the top variance
        vx = np.var(X,axis=0)
        which_x = np.argsort(vx)[::-1][:which_x]

    def proc_which_(which_la):
        if type(which_la) is int:
            vla = np.mean(np.var(La,axis=0),axis=1)
            return [(i, range(S)) for i in np.argsort(vla)[::-1][:which_la]]
        else:
            return which_la

    which_la = proc_which_(which_la)
    which_mu = proc_which_(which_mu)
    nx  = len(which_x)
    nla = len(which_la)
    nmu = len(which_mu)

    ntop = max([nx, nla, nmu])

    if draw_mode == "tall":
        nrows = ntop + 2 + 2
        ncols = 3
    else:
        nrows = ntop
        ncols = 6
        

    gs = GridSpec(nrows, ncols)

    colors = {"x":"black",              
              "la":"red",
              "mu":"blue"}
              
    # PLOT THE GC ACTIVITY

    Var = X if x_or_v == "x" else V
    yr = np.array([np.min(Var), np.max(Var)])
    yrm = np.mean(yr)
    yre = (yr - yrm)*1.1 + yrm
    
    for i,ix in enumerate(which_x):
        plt.subplot(gs[i,0])
        plt.plot(t, Var[:,ix],color=colormaps.rainbow(ix/float(Var.shape[0]))),
        plt.xticks([])
        plt.ylabel("#{}".format(ix))        
        #plt.ylim(yre)
        if i == 0:
            plt.title(x_or_v.upper())

    if draw_mode == "tall":
        plt.subplot(gs[ntop:(ntop+4), 0])
    else:
        plt.subplot(gs[:ntop, 3])
        
    v = np.var(Var, axis=0)
    iv = np.argsort(v)
    nv = 10

    for i in range(Var.shape[1]-nv, Var.shape[1]):
        x = Var[:,iv[i]]
        a = np.var(x)/(max(v) + 1e-6)
        plt.plot(t,Var[:,iv[i]]*100 + iv[i], alpha = a,
                 color = colormaps.rainbow(iv[i]/float(Var.shape[1])),
        )
        plt.xticks(np.arange(t[0],t[-1],0.1))
        plt.ylim(0,Var.shape[1]+100*np.max(Var))
            

    def plot_sister_activity_(La, which_la, column, cm_line, ttl):
        Lan = La/(np.max(np.abs(La))+1e-6) if normalize else La

        for i, ila_group in enumerate(which_la):
            plt.subplot(gs[i, column])
            iglom = ila_group[0]
            for which_sis in ila_group[1]:
                plt.plot(t, Lan[:,iglom, which_sis], color=cm_line(float(which_sis)/S))
                if normalize:
                    plt.ylim([-1,1])
            plt.ylabel("#{}".format(iglom))
            #plt.xticks([])
            plt.xticks(np.arange(t[0],t[-1],0.2))
            if i == 0:
                plt.title(ttl)

        if draw_mode == "tall":
            plt.subplot(gs[ntop:(ntop+4),column])
        else:
            plt.subplot(gs[:ntop, 3 + column])

        La1 = np.reshape(La,(La.shape[0], La.shape[1]*La.shape[2])).T

        plt.matshow(La1, fignum=False, aspect="auto", cmap=colormaps.seismic,
                    vmin=-np.max(np.abs(La)), vmax=np.max(np.abs(La)),
                    extent=[t[0],t[-1],0,La.shape[0]])

        plt.xticks(np.arange(t[0],t[-1],0.1))

        plt.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True,labeltop=False)

    plot_sister_activity_(La, which_la, 1, colormaps.rainbow, "La/max(|La|)" if normalize else "La")
    plot_sister_activity_(Mu, which_mu, 2, colormaps.rainbow, "Mu/max(|Mu|)" if normalize else "Mu")
    plt.tight_layout()
    

def multi_plot(y, x = None, wide = False):
    n = len(y)
    if x is None:
        x = np.arange(len(y[0]))
        
    if wide:
        plt.figure(figsize=(4*n,3))
        ax = [plt.subplot(1,n,i+1) for i in range(n)]
    else:
        plt.figure(figsize=(8,3*n))
        ax = [plt.subplot(n,1,i+1) for i in range(n)]

    for ax, yi in zip(ax,y):
        plt.sca(ax)        
        plt.plot(x, yi)
        plt.grid(True)
    plt.tight_layout()
    
def compare_abs_rel(true, pred, inds, wide = False):
    abs_errs = []
    norms    = []
    
    for i in range(len(inds)):
        norms.append(np.norm(true[i]))
        abs_errs.append(np.norm(true[i]-pred[i]))
    
    rel_errs  = abs_errs/np.array(norms)
    worst_abs = np.argmax(abs_errs)
    worst_rel = np.argmax(rel_errs)

    if wide:
        plt.figure(figsize=(16,3))
        ax = [plt.subplot(1,2,i) for i in range(1,3)]
    else:
        plt.figure(figsize=(8,6))
        ax = [plt.subplot(2,1,i) for i in range(1,3)]
        
    plt.sca(ax[0])
    plt.plot(t, true[worst_abs])
    plt.plot(t, pred[worst_abs])
    plt.title("Worst absolute at {} ({:1.3e})".format(inds[worst_abs], max(abs_errs)))
    plt.xlabel("t / sec")
    plt.ylabel("$\lambda_i(t)$")
    plt.grid(True)
    plt.sca(ax[1])
    plt.plot(t, true[worst_rel])
    plt.plot(t, pred[worst_rel])
    plt.title("Worst relative at {} ({:1.3e})".format(inds[worst_rel], max(rel_errs)))
    plt.xlabel("t / sec")
    plt.ylabel("$\lambda_i(t)$")
    plt.grid(True)
    plt.tight_layout()
    return abs_errs, rel_errs


def peak_freqs(x, fs):
    f = np.fft.fft(x);
    freqs = np.arange(len(x))/len(x)*fs   
    amp   = abs(f)[freqs<=fs/2]
    freqs = freqs[freqs<=fs/2]
    ind_peaks, *_ = find_peaks(amp)
    return [(freqs[i], amp[i]) for i in ind_peaks]

def euler(A, x0, dt, nt):
    x = np.zeros((nt, len(x0)))
    x[0] = x0
    for i in tqdm(range(1,nt)):
        x[i] = x[i-1] + dt * np.dot(A,x[i-1])
    return x.T
