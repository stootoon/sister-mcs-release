import os, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder",  help = "Folder to process.",    default = ".")
parser.add_argument("--eps",     help = "Solver eps parameter.", default = 5e-13)
args = parser.parse_args()

import numpy as np
import datatools as dt
import time

def proc1(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Could not find folder {folder}.")

    x_MAP_file = os.path.join(folder, "x_MAP.npy")
    if os.path.exists(x_MAP_file):
        print(f"{x_MAP_file=} exists, skipping.")
        return
    
    base_folder, params_folder = os.path.split(folder)

    # Look for the actual parameters file
    params_file = params_folder+".json"
    if not os.path.isfile(os.path.join(base_folder, params_file)):
        raise FileNotFoundError(f"Could not find file {params_file}.")

    # Get the default parameters and update from the file.
    params = dt.load_default_params()
    params.update(dt.load_params_from_file(base_folder, params_file))

    out_file = os.path.join(base_folder, params_folder+".cvx")
    with open(out_file, "w") as f:
        pass

    def echo(s):
        with open(out_file, "a") as f:
            f.write(s + "\n")
        print(s)

    echo(os.path.join(base_folder, params_file))
    echo(f"Running solver with eps={args.eps:1.3e}.")

    start_time = time.time()    
    x = None
    x_input_file = os.path.join(base_folder, params_file[:-5], "x_input.npy")
    if os.path.isfile(x_input_file):
        echo(f"Loading x_input from {x_input_file}.")
        x = np.load(x_input_file)
    else:
        echo(f"Could not find {x_input_file}, using x_true.")
    sol  = dt.get_x_MAP_for_params(params, x = x, silent_ob = True, full_sol = True, eps=args.eps)

    x_MAP= sol["x"]

    if sol["status"] != "optimal":
        echo("ERROR: Solution status '{}' was not 'optimal.'".format(sol["status"]))
    else:
        echo("Solution status was 'optimal'.")

    end_time   = time.time()
    elapsed    = end_time - start_time    
    echo(f"Solved in {elapsed:.3f} seconds.")        

    echo(f"Writing solution to {x_MAP_file}.")
    np.save(x_MAP_file, x_MAP)

    echo("Done.")
        
if "param" in args.folder:
    proc1(args.folder)
else:
    for item in os.listdir(args.folder):
        full_path = os.path.join(args.folder, item)
        if os.path.isdir(full_path) and "param" in item:
            print(f"\nProcessing {full_path}")
            proc1(full_path)

print("ALLDONE")
