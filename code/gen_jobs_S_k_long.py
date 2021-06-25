import os, sys
import pickle
import json
import itertools

header = """#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=JOBNAME
#SBATCH --ntasks=1
#SBATCH --time=6:00:00
#SBATCH --mem-per-cpu=32G
#SBATCH --partition=cpu

#SWEEP_PARAM PNAME  S, k
#SWEEP_PARAM PVALUE _pvalue_
#SWEEP_PARAM PRUN   _prun_

ml purge > /dev/null 2>&1
python -u ../run_sisters.py PARAMSFILE --write_every 100
"""

import datatools as dt

default_params = dt.load_default_params()
default_params["leak_pg"]   = 0
default_params["dt"]        = 1e-6
default_params["t_end"]     = 3
default_params["keep_till"] = 3

S_vals = [8]

sweep_name = f"sweep_S_k_long"
os.system("mkdir -p {}".format(sweep_name))

k_vals = [1,2,3,4,5,6,7,8,9,10]
n_runs = 12

job_id = 0
for S, k in itertools.product(S_vals, k_vals):
    for run in range(n_runs):
        params = dict(default_params)
        params["S"]    = S
        params["k"]    = k
        params["seed"] = job_id
        
        job_name = "S{}k{}_{}".format(S, k,run)
        job_script = header.replace("JOBNAME", job_name).replace("_pvalue_", "{}, {}".format(S, k)).replace("_prun_", str(run)).replace("PARAMSFILE", "params{}.json".format(job_id))
        
        params_file = os.path.join(sweep_name, "params{}.json".format(job_id))
        with open(params_file, "w") as out_file:
            json.dump(params, out_file)

        job_file = os.path.join(sweep_name, "job{}.sh".format(job_id))
        with open(job_file, "w") as out_file:
            out_file.write(job_script)

        print("Wrote {}.".format(job_file))
        
        job_id+=1
        

