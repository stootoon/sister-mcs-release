import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--njobs",   type=int, default=0, help = "Desired number of jobs.")
parser.add_argument("--nperjob", type=int, default=1, help="Number of parameters to run per job.")
parser.add_argument("--k",       type=int, default=3, help="Density of input odour.")
parser.add_argument("--nruns",   type=int, default=20, help="Number of runs per setting.")
args = parser.parse_args()

import os, sys
import pickle
import json
from copy import deepcopy
from random import shuffle

header = """#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=JOBNAME
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=cpu

#SWEEP_PARAM PNAME  _pname_
#SWEEP_PARAM PVALUE _pvalue_
#SWEEP_PARAM PRUN   _prun_

ml purge > /dev/null 2>&1
python -u ../run_sisters.py PARAMSFILE --write_every 100
"""
import datatools as dt

default_params = dt.load_default_params()
default_params["leak_pg"] = 0
default_params["dt"]      = 1e-6
default_params["k"]       = args.k
default_params["keep_till"] = 0.61
sweep_params = ["be", "sd", "ga", "tau_mc", "tau_gc", "tau_pg"]
factors = [0.5, 0.7071, 1, 1.414, 2]

sweep_name = "sweep_all_but_k{}".format(default_params["k"])
os.system("mkdir -p {}".format(sweep_name))

n_runs = args.nruns

S_vals = [1,2,4,5,6,8,10,12,16,20,24,25]

all_params = []

seed = 0

already_created_default_case = {(S,r):False for S in S_vals for r in range(n_runs)}
for param in sweep_params:
    print("*"*80)
    print("Parameter = {}".format(param))
    for factor in factors:
        print(" Factor = {}".format(factor))
        for S in S_vals:
            print("  S = {}".format(S))
            for run in range(n_runs):
                if factor == 1 and already_created_default_case[(S,run)] == True:
                    # Don't run the default case again
                    print("   Already created default case for S = {}, run = {}. Skipping.".format(S, run))
                    break
                params = deepcopy(default_params)
                params["param"] = param
                params[param] *= factor
                params["seed"] = run # Set the seed to be the run number so we can compare across different parameters 
                params["S"]    = S

                fname = "params{}.json".format(seed)
                params_file = os.path.join(sweep_name, fname)
                with open(params_file, "w") as out_file:
                    json.dump(params, out_file)

                all_params.append((seed, fname, params))

                seed+=1

                if factor == 1:
                    already_created_default_case[(S,run)] = True

shuffle(all_params)
job_id = 0
nperjob = args.nperjob if args.njobs<=0 else int(len(all_params)//args.njobs) + 1
for i in range(0,len(all_params),nperjob):
    files = [a[1] for a in all_params[i:i+nperjob]]
    job_name = "a{}".format(job_id)
    job_script = header.replace("JOBNAME", job_name).replace("_pname_", param).replace("_pvalue_", "{}".format(params[param])).replace("_prun_", str(run)).replace("PARAMSFILE", ",".join(files))
    job_file = os.path.join(sweep_name, "job{}.sh".format(job_id))
    with open(job_file, "w") as out_file:
        out_file.write(job_script)
    
    print("Wrote {}.".format(job_file))
    job_id += 1
print(f"Created {job_id} total jobs.")
print("DONE")

                

