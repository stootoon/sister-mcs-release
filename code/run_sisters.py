import os
import json
import argparse
import numpy as np
import datatools as dt

parser = argparse.ArgumentParser()
parser.add_argument("inputfiles",         help = "Comma separated list of json files containing the parameters .")
parser.add_argument("--write_every",      help = "The output stepsize to write.", type = int, default=1)
parser.add_argument("--report_frequency", help = "How often to report progress.", type = int, default=10)
args = parser.parse_args()

for paramsfile in args.inputfiles.split(","):
    paramsfile = paramsfile.strip()

    print("\nSTARTED {}".format(paramsfile))

    with open(paramsfile, "r") as in_file:
        p = json.load(in_file)

    for f in sorted(p):
        print("{:>16s}: {}".format(f, p[f]))

    res = dt.create_and_run_olfactory_bulb(p, keep_every = args.write_every, report_frequency = args.report_frequency)

    output_root = paramsfile.split(".json")[0]
    os.system("mkdir -p {}".format(output_root))
    
    summary = {f:res[f] for f in ["status", "last_iter"]}
    with open(os.path.join(output_root, "summary.json"), "w") as out_file:
        json.dump(summary, out_file)

        
    for var in ["T", "La", "Mu", "Y", "X", "V", "t_final", "la_final", "mu_final", "x_final", "v_final", "A10", "x_input"]:
        if var in res:
            v = res[var]    
            out_file = os.path.join(output_root, var + ".npy")
            np.save(out_file, v)
            print("Wrote {}.".format(out_file))
        else:
            print(f"Warning: did not find field {var} in results.")
        
print("ALLDONE.")

