import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--nperjob", type=int, default=1, help="Number of parameters to run per job.")
parser.add_argument("--nruns",   type=int, default=5, help="Number of runs per setting.")
args = parser.parse_args()

import os, sys
import pickle
import json
from copy import deepcopy
from random import shuffle
from itertools import product

header = """#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=JOBNAME
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=cpu

ml purge > /dev/null 2>&1
python -u ../run_sisters.py PARAMSFILE --write_every 100
"""

import datatools as dt

default_params = dt.load_default_params()
default_params["dt"]        = 1e-6
default_params["t_end"]     = 1.1
default_params["keep_till"] = 0.6 #1.1
default_params["leak_pg"]   = 0
default_params["spread"]    = 0.4
sweep_params = {"M":[50,100,200],
                "N":[1200,2400,4800],
                "k":[1,3,5,10],
                "S":[25], #[1,2,4,8,25],
                "run":list(range(1, args.nruns+1))}

sweep_name = "sweep_MNk_conc_spread"
os.system("mkdir -p {}".format(sweep_name))

pid = 0
all_params = []

flds = sorted(list(sweep_params.keys()))
vals = [sweep_params[f] for f in flds]
for v1 in product(*vals):
    params = deepcopy(default_params)
    for f,v in zip(flds, v1):
        params[f if f != "run" else "seed"] = v # Set the seed to be the run number so we can compare across different parameters 

    fname = "params{}.json".format(pid)
    params_file = os.path.join(sweep_name, fname)
    with open(params_file, "w") as out_file:
        json.dump(params, out_file)

    all_params.append((pid, fname, params))

    pid+=1

shuffle(all_params)

job_id = 0
for i in range(0,len(all_params),args.nperjob):
    files = [a[1] for a in all_params[i:i+args.nperjob]]
    job_name   = "a{}".format(job_id)
    job_script = header.replace("JOBNAME", job_name).replace("PARAMSFILE", ",".join(files))
    job_file   = os.path.join(sweep_name, "job{}.sh".format(job_id))
    with open(job_file, "w") as out_file:
        out_file.write(job_script)
    
    print("Wrote {}.".format(job_file))
    job_id += 1
                    

                

