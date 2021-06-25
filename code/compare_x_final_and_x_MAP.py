import os, sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder",  help = "Folder to process.", default = ".")
parser.add_argument("--verbose", help = "Whether to print out the comparison details.", default=False)
args = parser.parse_args()

import numpy as np
import datatools as dt

def proc1(folder):
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Could not find folder {folder}.")
    
    x_final_file = os.path.join(folder, "x_final.npy")
    if not os.path.isfile(x_final_file):
        raise FileNotFoundError(f"Could not find {x_final_file}.")

    x_MAP_file = os.path.join(folder, "x_MAP.npy")
    if not os.path.isfile(x_MAP_file):
        raise FileNotFoundError(f"Could not find {x_MAP_file}.")
    
    base_folder, params_folder = os.path.split(folder)
    out_file = os.path.join(base_folder, params_folder+".cmp")
    if os.path.exists(out_file):
        print(f"{out_file=} exists, skipping.")
        return
    
    # Look for the actual parameters file
    params_file = params_folder+".json"
    if not os.path.isfile(os.path.join(base_folder, params_file)):
        raise FileNotFoundError(f"Could not find file {params_file}.")

    # Get the default parameters and update from the file.
    params = dt.load_default_params()
    params.update(dt.load_params_from_file(base_folder, params_file))
    
    with open(out_file, "w") as f:
        pass

    def echo(s):
        with open(out_file, "a") as f:
            f.write(s)
        print(s)

    # The comparison makes sense only when leak_pg = 0
    if params["leak_pg"] != 0:
        echo("leak_pg = {} != 0, skipping.".format(params["leak_pg"]))
        return
    
    obj_fun   = dt.get_loss_function(base_folder, params_file)
    y         = np.load(os.path.join(folder, "Y.npy"))[-1,:,-1] 
    x_MAP     = np.load(x_MAP_file)
    x_final   = np.load(x_final_file)
    
    loss_x_MAP   = obj_fun(y, x_MAP)
    loss_x_final = obj_fun(y, x_final)
    norm_dx      = np.linalg.norm(x_MAP-x_final)
    diff_loss    = loss_x_MAP - loss_x_final
    
    if args.verbose is True:
        echo(os.path.join(base_folder, params_file))    
        echo(f"            |x_MAP - x_final|: {norm_dx:1.3e}")
        echo(f"               Loss (x_MAP)  : {loss_x_MAP}")
        echo(f"               Loss (x_final): {loss_x_final}")
        echo(f"Loss (x_MAP) - Loss (x_final): {diff_loss}")
    else:
        echo(f"{diff_loss}")
        
if "param" in args.folder:
    proc1(args.folder)
else:
    for item in os.listdir(args.folder):
        full_path = os.path.join(args.folder, item)
        if os.path.isdir(full_path) and "param" in item:
            print(f"Processing {full_path}")
            proc1(full_path)

print("ALLDONE")
