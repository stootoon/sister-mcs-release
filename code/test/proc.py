import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from numpy import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--rootdir", help = "Root directory to read fields from", default="params")
args = parser.parse_args()
print(f"Reading fields from {args.rootdir}")

fields = ["T", "La", "Mu", "Y", "X", "V", "t_final", "la_final", "mu_final", "x_final", "v_final"]
data = {}
for fld in fields:
    file_name = os.path.join(args.rootdir, fld+".npy")
    print(f"Loading {file_name}")
    data[fld] = np.load(file_name)
    print(data[fld].shape)

fig = plt.figure(figsize=(8,8))
gs = GridSpec(4,2)
m = 5
print(f"Plotting data for glomerulus {m}")
for i, fld in enumerate(["La", "Mu", "X", "V"]):
    plt.subplot(gs[i,0])
    if fld in ["La","Mu"]:
        plt.plot(data["T"], data[fld][:,m,:])
    else:
        plt.plot(data["T"], data[fld])
    plt.xlabel("Time (sec.)")
    plt.ylabel(fld)

    plt.subplot(gs[i,1])
    fldf = fld.lower() + "_final"
    if fld in ["La","Mu"]:
        plt.plot(data[fldf][m,:])
    else:
        plt.plot(data[fldf])

    t_final = data["t_final"]
    plt.title(f"{fld} at t = {t_final}")
plt.tight_layout()
plt.show()
fig.savefig("proc.pdf", bbox_inches="tight")
print("Done.")
    
    

