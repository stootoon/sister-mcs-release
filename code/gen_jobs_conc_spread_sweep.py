# Here we're generating jobs for the condition where there's a spread of concentrations values.
# This is mainly to demonstrate convergence and effects of leak more clearly for the k>1 cases.

import os, sys
import pickle
import json
import itertools
import numpy as np

header = """#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=JOBNAME
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=64G
#SBATCH --partition=cpu

#SWEEP_PARAM PNAME  S, k
#SWEEP_PARAM PVALUE _pvalue_
#SWEEP_PARAM PRUN   _prun_

ml purge > /dev/null 2>&1
python -u ../run_sisters.py PARAMSFILE --write_every 100
"""

import datatools as dt

default_params = dt.load_default_params()
default_params["dt"]        = 1e-6
default_params["keep_till"] = 0.6
default_params["t_end"]     = 2.1
default_params["spread"]    = 0.4

sweep_name = "sweep_conc_spread"
os.system("mkdir -p {}".format(sweep_name))

S_vals    = [60,80] + [1,2,8,25]
#leak_vals = [0,1,10,100]
leak_vals = list(np.arange(0.1,1,0.1)) + [1.5, 2, 3]
k_vals    = [1,3,10]
n_runs    = 5

job_id = 1000
for S, leak, k in itertools.product(S_vals, leak_vals, k_vals):
    for run in range(n_runs):
        params = dict(default_params)
        params["S"] = S
        params["k"] = k
        params["leak_pg"] = leak
        params["seed"] = run
        
        job_name = f"{S}S{leak}L{run}"
        job_script = header.replace("JOBNAME", job_name).replace("_pvalue_", "{}, {}".format(S, leak)).replace("_prun_", str(run)).replace("PARAMSFILE", "params{}.json".format(job_id))
        
        params_file = os.path.join(sweep_name, "params{}.json".format(job_id))
        with open(params_file, "w") as out_file:
            json.dump(params, out_file)

        job_file = os.path.join(sweep_name, "job{}.sh".format(job_id))
        with open(job_file, "w") as out_file:
            out_file.write(job_script)

        print("Wrote {}.".format(job_file))
        
        job_id+=1
        

