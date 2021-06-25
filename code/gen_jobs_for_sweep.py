import argparse
import itertools

parser = argparse.ArgumentParser()
parser.add_argument("sweep_file",    type=str, help="JSON file containing the sweep parameter.")
parser.add_argument("--nruns",       type=int, default=10, help="Number of runs per parameter.")
parser.add_argument("--nperjob",     type=int, default=1,  help="Number of parameters to run per job.")
parser.add_argument("--mem", type=str, default="16G",      help="Memory per cpu.")
parser.add_argument("--time",        type=str, default="24:00:00", help="Time for job.")
parser.add_argument("--seed-from-run", action="store_true", help="Whether to set the seed value from the run value. Useful for keeping the seed the same while changing a parameter.")
args = parser.parse_args()
print(args)

import os, sys
import pickle
import json
from copy import deepcopy
from random import shuffle

header = """#!/bin/bash
# Simple SLURM sbatch example
#SBATCH --job-name=JOBNAME
#SBATCH --ntasks=1
#SBATCH --time=TIME
#SBATCH --mem-per-cpu=MEM
#SBATCH --partition=cpu

#PARAM_NAMES  _pnames_
#PARAM_VALUES _pvalues_

ml purge > /dev/null 2>&1
python -u ../code/run_sisters.py PARAMSFILE --write_every 100
""".replace("TIME", args.time).replace("MEM", args.mem)

sweep_name = args.sweep_file.split(".json")[0]
with open(args.sweep_file, "r") as f:
    params = json.load(f)

params["run"] = list(range(args.nruns))


print(sweep_name)
os.system("mkdir -p {}".format(sweep_name))

sweep_param_names = [k for k,p in params.items() if type(p) is list]
header = header.replace("_pnames_", ",".join(sweep_param_names))
print(header)

params_as_lists = {k:p if type(p) is list else [p] for k,p in params.items()}

fields = list(params_as_lists.keys())
values = [params_as_lists[f] for f in fields]
single_params = [{fields[i]:val[i] for i in range(len(fields))} for val in list(itertools.product(*values))]

print("Generated {} parameters.".format(len(single_params)))
all_params = []
for i, params in enumerate(single_params):
    seed = params["run"] if args.seed_from_run is True else i
    params["seed"] = seed
    fname = "params{}.json".format(i)
    params_file = os.path.join(sweep_name, fname)
    with open(params_file, "w") as out_file:
        json.dump(params, out_file)

    all_params.append((seed, fname, params))

shuffle(all_params)
print(all_params[0])

job_id = 0
for i in range(0,len(all_params),args.nperjob):
    files = [a[1] for a in all_params[i:i+args.nperjob]]
    job_name = "a{}".format(job_id)
    parami = [all_params[ii][2] for ii in range(i,i+args.nperjob)]
    pvalues = "{}".format([[p[f] for f in sweep_param_names] for p in parami])
    job_script = header.replace("JOBNAME", job_name).replace("_pvalues_", pvalues).replace("PARAMSFILE", ",".join(files))
    job_file = os.path.join(sweep_name, "job{}.sh".format(job_id))
    with open(job_file, "w") as out_file:
        out_file.write(job_script)
    
    print("Wrote {}.".format(job_file))
    job_id += 1
                    

                

