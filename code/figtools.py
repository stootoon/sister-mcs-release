import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import cm
from matplotlib import colors as mcolors
from matplotlib.transforms import TransformedBbox, Bbox
import matplotlib.ticker as ticker
from matplotlib import cycler
from cycler import Cycler

fig_params = {"width":12, "height":3, "fontsize":10, "legend_fontsize":8}

import colorsys

def apply_styles():
    plt.style.use("default")
    plt.style.use("ms.mplstyle")

cm_sub = lambda cm,h0,h1: lambda r: cm(r*(h1 - h0) + h0)

hex2rgba = mcolors.to_rgba
set_alpha = lambda a: lambda col: list(col[:3]) + [a]
alpha_fun = set_alpha(0.5)

def color2hsv(rgb):
    if type(rgb) is str:
        rgb = hex2rgba(rgb)
    h,s,v  = colorsys.rgb_to_hsv(*rgb[:3])
    return h,s,v

def set_sat(rgb, sat):
    h,s,v  = color2hsv(rgb)
    rgb = colorsys.hsv_to_rgb(h,sat,v)
    return rgb

def change_hue(rgb, amt):
    h,s,v  = color2hsv(rgb)    
    rgb    = colorsys.hsv_to_rgb(np.mod(h + amt,1),s,v)
    return rgb

def change_saturation(rgb, new_sat):
    h,s,v  = color2hsv(rgb)    
    rgb    = colorsys.hsv_to_rgb(h,new_sat,v)
    return rgb

def color_for_param(center_color, ip, start = 1/2, end = 1):
    # ip should be between -1 and 1
    # 0  for the center value    
    h,s,v = color2hsv(center_color)
    delta = end - start
    new_sat = start + (ip-(-1))/2*delta
    return change_saturation(center_color, new_sat)

def demo_colors(colors):
    for i,col in enumerate(colors):
        plt.plot([0,1],[-i,-i], color=col, linewidth=12, label = str(i))
    plt.gca().set_xticks([])
    plt.gca().set_yticks(-np.arange(0,len(colors)))
    plt.gca().set_yticklabels(map(str,np.arange(0,len(colors))))
    plt.gca().spines["bottom"].set_visible(False)    
    plt.axis("tight")

yticklabs = lambda s: gca().set_yticklabels(s)
xticklabs = lambda s: gca().set_xticklabels(s)
xticks    = lambda s: gca().xaxis.set_ticks(s)
yticks    = lambda s: gca().yaxis.set_ticks(s)

def gen_hue_spread_colors(central_color, n_cols, hue_spread = 0.05, alpha = 1.):
    hues = np.linspace(-hue_spread/2, hue_spread/2, n_cols)
    cols = [ set_alpha(alpha)(change_hue(central_color, h)) for h in hues]
    return cols

def gen_gc_cols(N=1200, ind_active=[300,600,900], central_color = "mediumspringgreen", hue_spread = 0.05, seed = 0, alpha_inactive = 0.5, alpha_active=0.75):
    np.random.seed(seed)
    cols = [set_alpha(alpha_inactive)(cm.Greys(np.random.rand()*0.8 + 0.2)) for i in range(N)]
    hues = np.linspace(-hue_spread/2, hue_spread/2, len(ind_active))
    for i, ind in enumerate(ind_active):
        cols[ind] = set_alpha(alpha_active)(change_hue(central_color, hues[i]))

    return cols

pop_cmaps = {"mc":cm_sub(cm.Blues,   0.3,1),
             "pg":cm_sub(cm.Purples, 0.3,1),
             "gc":cm_sub(cm.Greens,  0.3,1)}

cyc_gc = cycler(color=gen_gc_cols(1200, [300,600,900], central_color = pop_cmaps["gc"](0.5)))

def Scols(pop, Smin=2, Smax=25):
    Sfull  = [1] + list(range(Smin,Smax+1))
    nS     = len(Sfull) - 1 # -1 to not include S = 1    
    cols_list = [pop_cmaps[pop]((i+1)/nS) for i in range(nS)]
    # Prepend the colors with gray to cover the S = 1 case
    cols_list = [[0.5]*3+[1]] + cols_list
    cols_dict = {S:cols_list[Sfull.index(S)] for S in Sfull}
    
    return cols_list, cols_dict
    
def hsl_to_rgb(hsl):
    h,s,l = hsl
    v = l + s*min(l, 1-l)
    s = 0 if v == 0 else 2*(1 - l/v)
    rgb = mcolors.hsv_to_rgb([h,s,v])
    return rgb

def hsl_cms(s = 0.5, l=0.5, n = 3, offset = 0,  width = 1., demo=True, n_per = 4, alpha = 1):
    # hue centers ∈ [0,1] 
    h = np.mod(np.arange(n)/n + offset, 1)
    w = 1/n*width
    cols = [[list(hsl_to_rgb([np.mod(hi + dh * w - w/2, 1), s, l])) + [alpha] for dh in np.arange(n_per)/(n_per-1)] for hi in h]
    # Gotta do it this complicated way because using the inner lambda causes closure problems
    cms = [(lambda hh: lambda dh: list(hsl_to_rgb([np.mod(hh + dh * w - w/2, 1), s, l])) + [alpha])(hi) for hi in h] 
        
    if demo:
        plt.figure(figsize=(8,4/3*n))
        t = np.linspace(0,1,1001)
        f = np.linspace(5,50,100)
        S = np.sin(np.outer(t, 2*np.pi*f))
        C = np.cos(np.outer(t, 2*np.pi*f))
        nf = len(f)
        for i in range(n):
            plt.subplot(n,1,i+1)
            for j in range(n_per):
                y = np.dot(S, np.random.randn(nf)) + np.dot(C, np.random.randn(nf))
                plt.plot(t,y,color=cols[i][j], linewidth=2)
                
    return cms
    

def plot0(t, X, ax = None, plot_fun = "plot", mode = "line", xlabel_args = {}, ylabel_args={}, title_args={}, col_cyc = None, ax_pre = None, plot_args = {}, *args, **kwargs):
    if ax is None:
        ax = plt.gca()
    
    plt.sca(ax)

    if ax_pre:
        ax_pre(ax)
    
    if col_cyc:
        if type(col_cyc) is Cycler:
            ax.set_prop_cycle(col_cyc)
        else:
            ax.set_prop_cycle(cycler(color=[col_cyc]))

    pf = eval("ax."+plot_fun)

    if mode == "scatter":
        h = pf(np.array(t).reshape((1,-1)), np.array(X).reshape((1,-1)), **plot_args)
    else:
        h = pf(t, X, **plot_args);

    labels = {"xlabel":(ax.set_xlabel, xlabel_args),
              "ylabel":(ax.set_ylabel, ylabel_args),
               "title":(ax.set_title,   title_args)}
    for k, v in kwargs.items():
        if k in labels:
            labels[k][0](v, **labels[k][1])
        elif k in ["xticks", "xticklabels", "yticks", "yticklabels"]:
            eval("ax.set_" + k)(v)

    # Set the lims last so they're not overriden
    for lim in ["xlim","ylim"]:
        if lim in kwargs:
            eval("ax.set_" + lim)(kwargs[lim])

    return ax, h

def label_axes(ax_list, labs, dx=0, dy=0, *args, **kwargs):
    fig = plt.gcf()
    renderer = fig.canvas.get_renderer()
    trans    = fig.transFigure
    itrans   = trans.inverted()
    for i, (ax, lab) in enumerate(zip(ax_list, labs)):
        bb = ax.get_tightbbox(renderer)
        bb = TransformedBbox(bb, itrans)
        #print(lab, bb.x0, bb.y1)
        dxi = dx[i] if hasattr(dx, "__len__") else dx
        dyi = dy[i] if hasattr(dy, "__len__") else dy
        fig.text(bb.x0 + dxi, bb.y1 + dyi, lab, *args, transform=trans, **kwargs)


def tighten_row(ax, fig, renderer = None, h_w = 0):
    # Tightens a row of axes in a figure so that there's no whitespace between them
    if renderer is None:
        renderer = fig.canvas.get_renderer()
    bb2p = lambda bb: [bb.x0,bb.x1,bb.y0,bb.y1]
    bb2w = lambda bb: bb.x1-bb.x0
    pos0 = lambda ax: ax.get_position()
    pos1 = lambda ax: TransformedBbox(ax.get_tightbbox(renderer), fig.transFigure.inverted())

    lb,rb = bb2p(pos1(ax[0]))[0], bb2p(pos1(ax[-1]))[1]
    # get the ylabel widths. These won't change
    ylw = [bb2w(pos1(a)) - bb2w(pos0(a)) for a in ax]
    aw = (rb - lb - sum(ylw) - h_w*(len(ax)-1))/len(ax) # Axis widths
    for i,a in enumerate(ax):
        old_pos = pos0(a)
        new_x0 = aw*i + sum(ylw[:i+1])
        new_x1 = new_x0 + aw
        new_pos = Bbox([[new_x0, old_pos.y0], [new_x1,old_pos.y1]])
        a.set_position(new_pos)
    
def set_params(**kwargs):
    global fig_params
    for k,v in kwargs.items():
        fig_params[k] = v

def sort_by_peak_time(X, signed = True):
    amax = np.max(abs(X), axis=0)
    imax = np.argmax(abs(X), axis=0)
    smax = np.array([np.sign(X[im,i]) for i,im in enumerate(imax)]) if signed else 1
    return np.argsort(imax*smax)

def sort_by_variance(X, signed = True):
    # sign by energy
    Epos = np.sum((X**2)*(X>0),axis=0)
    Eneg = np.sum((X**2)*(X<0),axis=0)
    s    = np.sign(Epos - Eneg)
    return np.argsort(np.var(X, axis=0)*s)

id2name = lambda S: lambda i: "({},{})".format(i//S, i % S)

def plot1(X, sort_function, t = []):
    nt, M, S = X.shape
    if t == []:
        t = np.arange(0,nt)
    lar = X.reshape(nt, -1)
    blues = cycler('color', [cm.Blues(i) for i in np.linspace(0.1,0.9,5)])
    reds  = cycler('color', [cm.Reds(i)  for i in np.linspace(0.1,0.9,5)])
    ind = sort_function(lar)
    plt.figure(figsize=(fig_params["width"],fig_params["height"]))
    plt.subplot(1,4,1); plt.matshow(lar[:,ind].T,vmin=-2,vmax=2, extent = [t[0], t[-1], 0, M*S], cmap=cm.seismic, fignum=False); plt.axis("normal")
    plt.subplot(1,4,2); plt.gca().set_prop_cycle(blues); plt.plot(t, lar[:, ind[:5]]);  plt.axis("normal"); plt.legend(list(map(id2name(S),ind[:5])))
    plt.subplot(1,4,3); plt.gca().set_prop_cycle(reds);  plt.plot(t, lar[:, ind[-5:]]); plt.axis("normal"); plt.legend(list(map(id2name(S),ind[-5:])))
    ind_r  = np.random.permutation(ind)[:5]
    plt.subplot(1,4,4); plt.plot(t,lar[:, ind_r]); plt.axis("normal"); plt.legend(list(map(id2name(S),ind_r)))
    plt.tight_layout()


def plot_pca(X, t = [], n_plot = 6, xlim = [], cmap=cm.winter):
    nt, M, S = X.shape
    if t == []:
        t = range(nt)
    if xlim == []:
        xlim = [t[0], t[-1]]
        
    lar = X.reshape(nt, -1)
    U,s,V = np.linalg.svd(lar)
    n_cols = n_plot // 2 + (n_plot % 2)
    gs = GridSpec(2, 1 + n_cols)
    plt.figure(figsize=(fig_params["width"], fig_params["height"]))
    plt.subplot(gs[0,0])
    vsub = V[:10,:]
    vext = np.max(np.abs(vsub))
    plt.matshow(V[:10,:], vmin=-vext,vmax=vext, cmap=cm.seismic, fignum=False)
    plt.xlabel("cell")
    plt.ylabel("coef")
    plt.axis("normal")

    plt.subplot(gs[1,0])
    plt.semilogx(range(1,len(s)+1), s, "r.-")
    plt.xlabel("component")
    plt.ylabel("value")

    cols = [cmap(f) for  f in np.linspace(0.1,0.9,n_plot)]
    for i in range(n_plot):
        plt.subplot(gs[i//n_cols, 1 + (i % n_cols)])
        plt.plot(t, U[:,i]*(2*(max(U[:,i])>max(-U[:,i])) - 1), color=cols[i], label="PC {}".format(i))
        plt.xlim(xlim)
        plt.legend()
    plt.tight_layout()

two_norm = lambda X, x_true: np.sqrt(np.sum((X - x_true[np.newaxis,:])**2,axis=1))

def frac_fp_energy(X, x_true):
    Xtp = X[:, np.abs(x_true)>1e-12]
    Xfp = X[:, np.abs(x_true)<1e-12]

    tp_energy = np.sum(Xtp**2, axis=1)
    fp_energy = np.sum(Xfp**2, axis=1)
    return fp_energy/(tp_energy + fp_energy + 1e-12)
    
    
def plot_error_timecourse(results, x_true, metric = two_norm, plotfun = plt.plot):
    X = results["x"]
    t = results["t"]
    err = metric(X, x_true)
    if plotfun is not None:
        plt.figure(figsize=(fig_params["width"], fig_params["height"]))
        plotfun(t,err)
        plt.xlabel("time")
        plt.ylabel("error")
    return err, t

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
        plt.plot(t, Var[:,ix],color=cm.rainbow(ix/float(Var.shape[0]))),
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
                 color = cm.rainbow(iv[i]/float(Var.shape[1])),
                 #color = named_colors["gray"] if x_true[iv[i]]<1e-3 else named_colors[colors["x"]]
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

        plt.matshow(La1, fignum=False, aspect="auto", cmap=cm.seismic,
                    vmin=-np.max(np.abs(La)), vmax=np.max(np.abs(La)),
                    extent=[t[0],t[-1],0,La.shape[0]])

        plt.xticks(np.arange(t[0],t[-1],0.1))

        plt.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True,labeltop=False)

    plot_sister_activity_(La, which_la, 1, cm.rainbow, "La/max(|La|)" if normalize else "La")
    plot_sister_activity_(Mu, which_mu, 2, cm.rainbow, "Mu/max(|Mu|)" if normalize else "Mu")
    plt.tight_layout()
