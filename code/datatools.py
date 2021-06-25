import os, json, re
import logging
import pandas as pd
import numpy as np
import json
from collections import namedtuple
from tqdm import tqdm
from itertools import product
from copy import copy, deepcopy
import olfactory_bulb
from olfactory_bulb import OlfactoryBulb, Odor, get_x_true
from scipy.signal import find_peaks, argrelextrema, find_peaks_cwt
from functools import partial
import pickle
import util
import pdb

log = util.create_logger("datatools")
INFO  = log.info
DEBUG = log.debug

log.setLevel(logging.INFO)

all_vars = ["la_final", "mu_final", "x_input", "x_final", "x_MAP", "t_final", "La", "Mu", "T", "V", "X", "Y"]

# Looks for files ending in VARNAME.npy and loads them into a dictionary.
def load_vars(data_root, params_dir, vars=all_vars, warn_on_missing = True):
    results = {}
    if vars is None:
        vars = all_vars
    for v in vars:
        full_file = os.path.join(data_root, params_dir, v + ".npy")
        if os.path.isfile(full_file):
            results[v] = np.load(full_file)
        else:
            if warn_on_missing:
                log.warning(f"Could not find {full_file}.")
    return results
        
load_results = lambda data_root, params_dir, vars = all_vars, warn_on_missing=True: [
    DEBUG(f"Loading " + "{}".format(all_vars if vars is None else vars) + f" from {data_root}/{params_dir}."),
    load_vars(data_root, params_dir, vars, warn_on_missing)][1]

data_folder = lambda subfolder: os.path.join(os.environ["SISTER_MCS_DATA"], subfolder)

def load_if_exists_else_compute(data_file, loader_fun, force_compute = False, save_after_compute = True):
    if os.path.isfile(data_file) and force_compute is False:
        INFO(f"Loading from {data_file}.")
        with open(data_file, "rb") as f:
            data = pickle.load(f)
        INFO(f"Done loading.")            
    else:
        if force_compute:
            INFO(f"force_compute was True, so computing data.")
        elif not os.path.isfile(data_file):
            INFO(f"Could not find data file {data_file}, so computing data.")
        else:
            raise ValueError("force_compute is False or file exists, shouldn't be computing!")
        data = loader_fun()
        if save_after_compute:
            INFO(f"Saving results to {data_file}.")
            with open(data_file, "wb") as f:
                pickle.dump(data, f)
        INFO(f"Done computing.")
    return data

class SilenceLoggers:
    def __init__(self, logger_names, silence_level = logging.WARNING):
        self.logger_names  = logger_names if type(logger_names) is list else [logger_names]
        self.logger_levels = {name:logging.getLogger(name).level for name in self.logger_names}
        self.silence_level   = silence_level

    def __enter__(self):
        [logging.getLogger(name).setLevel(self.silence_level) for name in self.logger_names]
        log.debug(f"Silenced loggers {self.logger_names} by setting level to {self.silence_level}.")

    def __exit__(self, *args):
        [logging.getLogger(name).setLevel(lev) for name,lev in self.logger_levels.items()]        
        log.debug(f"Reset loggers to original levels: {self.logger_levels}.")
    
def hashify_dict(d):    
    def _hashify(f):
        if type(f) is list:
            return tuple(_hashify(fi) for fi in f)
        elif type(f) is dict:
            return tuple((k, _hashify(f[k])) for k in sorted(f))
        else:
            return f

    return {k:_hashify(v) for k, v in d.items()}

def compare_parameters(p0, p1, mismatch_missing_fields = False):
    match = True
    k0 = set(list(p0.keys()))
    k1 = set(list(p1.keys()))
    if len(k0-k1):
        INFO(f"Parameter set 0 has fields {k0-k1} not in parameter set 1.")
        match = False if mismatch_missing_fields else match
    elif len(k1-k0):
        INFO(f"Parameter set 1 has fields {k1-k0} not in parameter set 0.")
        match = False if mismatch_missing_fields else match
    else:
        INFO(f"Both parameter sets have the same fields. OK.")
    for k in (k0 & k1):
        if p0[k] != p1[k]:
            INFO(f"p0[{k}] = {p0[k]} does not equal p1[{k}] = {p1[k]}.")
            match = False
    if match is True:
        INFO(f"Parameter sets matched.")
    else:
        INFO(f"Parameter sets DO NOT match.")
    return match
        

def load_params_from_folder(root_folder, mark_every = 500):
    """Walks the root_folder, reads all parameter json files, 
    and loads the results into a dataframe. 
    
    RETURNS   
    df: The dataframe containing all the parameters read.
    params: A list of dictionaries containing the parameters
            read.
    """
    if not os.path.isdir(root_folder):
        raise FileNotFoundError(f"Could not load params because folder '{root_folder}' does not exist.")
    params = []
    count = 0
    for root, dirs, files in os.walk(root_folder):
        for f in files:
            if re.match("params[0-9]+\.json", f):
                full_file = os.path.join(root, f)
                with open(full_file, "r") as in_file:
                    # Hashify so that we can then form set operations on them.
                    # We need this in e.g. FreeSweepDataset to see how many values each field has.
                    new_p = hashify_dict(json.load(in_file))
                    new_p["file"] = f
                            
                    params.append(new_p)
                count += 1
                if (count % mark_every) == 0:
                    INFO(f"Read params files #{count:<5d} in {root_folder}...")
    INFO(f"Read {count} params files in {root_folder}.")                    
    df = pd.DataFrame(params)
    return df, params

def load_params_from_file(root_folder, params_file):
    """Loads the parameters from the json file root_folder/params_file
    and returns them as a dictionary."""
    params_file = params_file if params_file.endswith("json") else params_file + ".json"
    full_file = os.path.join(root_folder, params_file)
    with open(full_file, "r") as in_file:
        p = json.load(in_file)
    return p

def load_default_params(root_folder = os.environ["SISTER_MCS"], file_name = "default_params.json"):
    """Loads the default parameter values from file 
    and returns them as a dictionary."""
    return load_params_from_file(root_folder, file_name)

def get_x_MAP_for_params_file(root_folder, params_file, y = None, silent_ob = True, **kwargs):
    """Computes the exact solution for the parameters
    in the specified file and returns 'x_MAP'. The odour 
    input 'y' is computed from the parameters unless 
    overriden by the keyword argument."""
    p = load_params_from_file(root_folder, params_file)
    return get_x_MAP_for_params(p, y=y, silent_ob=silent_ob, **kwargs)

def get_x_MAP_for_params(p, y = None, x = None, silent_ob = True, full_sol = False, return_y = False, **kwargs):
    if silent_ob:
        ob_logger_level = olfactory_bulb.logger.level
        olfactory_bulb.logger.setLevel(logging.WARNING)
    OB = OlfactoryBulb(**p)
    if y is None:
        if x is None:
            x = np.array(get_x_true(p["N"], p["k"], spread=p["spread"]))
        y = np.dot(OB.A, x)

    sol     = OB.run_exact(y, **kwargs)
    sol_ret = sol if full_sol is True else sol["x"]
    
    if silent_ob:
        olfactory_bulb.logger.setLevel(ob_logger_level)
    
    return (sol_ret, y) if return_y is True else sol_ret
    
def get_matrix_for_params(root_folder, params_file):
    """Returns the affinity matrix for the specified 
    parameters file."""
    p = load_params_from_file(root_folder, params_file)
    with SilenceLoggers("olfactory_bulb"):
        OB = OlfactoryBulb(**p)
    return OB.A

def get_perp(A):
    M,n = A.shape
    m = M - n
    AP = np.random.randn(M,M)
    AP[:,:n] = A
    for m in range(n, M):
        B = AP[:,:m]
        v = AP[:,m]
        H = np.eye(M) - np.linalg.multi_dot((B, np.linalg.inv(np.dot(B.T, B)), B.T))
        AP[:,m] = np.dot(H,v)
    AP = AP[:, n:]
    AP /= np.linalg.norm(AP,axis=0)[np.newaxis,:]     
    return AP

def get_loss_function(root_folder, file_name):
    """Returns the loss function L(y,x) for the specified parameters."""
    p = load_params_from_file(root_folder, file_name)
    A = get_matrix_for_params(root_folder, file_name)
    be,ga,sd = p["be"], p["ga"], p["sd"]
    return lambda y,x: be*sum(abs(x)) + ga*sum(x**2)/2 + sum((y - np.dot(A,x))**2)/2/sd/sd

df_Sk = []
SkResults = namedtuple('SkResults',['results_array','t','params_files'])
def load_Sk(S, k, root_folder = "sweep_S_k", vars_to_load=["X", "La", "Y", "Mu", "x_final", "la_final", "mu_final", "t_final", "x_MAP"], n_max = 100, force_reload = False):
    """Loads all results for the specified S and k values, and for the specified variables.

    RETURNS
    data: A list of dictionaries containing the results for each instance found.
    t: The time vector.
    param_files: The list of parameter files corresponding to the elements of data."""

    global df_Sk
    folder = data_folder(root_folder)
    INFO(f"Loading sweep_S_k from {folder}")
    if len(df_Sk) == 0 or force_reload == True: 
        INFO("Loading df_Sk")
        df_Sk, _ = load_params_from_folder(folder)

    # df_Sk has rows for all S and k. Subset to the ones we want.
    df = df_Sk[(df_Sk["S"]==S) & (df_Sk["k"]==k)]
    params_files = list(df["file"])
    INFO("Found %d files for S = %d, k = %d.", len(params_files), S, k)
    if "x_final" not in vars_to_load:
        vars_to_load.append("x_final")
    if "la_final" not in vars_to_load:
        vars_to_load.append("la_final")
    if "x_MAP" not in vars_to_load:
        vars_to_load.append("x_MAP")
        
    files_to_load = params_files[:n_max]
    INFO(f"Loading first {len(files_to_load)} files.")
    # Load the time index, and make sure they're all the same
    INFO(f"Loading time index from the first file.")    
    t = load_results(folder, files_to_load[0][:-5], vars=["T"])["T"]
    INFO(f"Checking against time indices of remaining files.")    
    for f in files_to_load[1:]:
        tf = load_results(folder, f[:-5], vars=["T"])["T"]
        if not np.allclose(t, tf):
            raise ValueError("Time indices in file {} did not match those in file {}".format(files_to_load[0], f))
    INFO(f"Time indices matched.")            
    # Now actually load the results
    INFO(f"Loading the actual results.")        
    results_array = [load_results(folder, f[:-5], vars=vars_to_load) for f in files_to_load]
    results_array = _post_load(results_array, files_to_load, folder)
    # Get the params files here because _post_load sorts the results by seed.
    # So they might not be in the order the files were found above
    params_files  = [item["file"] for item in results_array]
    return SkResults(results_array=results_array, t=t, params_files=params_files)

def _post_load(data, params_files, root_folder):
    """Append a few additional fields to each element of the
    list of dictionaries data and return it.
    
    RMSE values are computed relative to the final value for 'la' and
    'mu', and against x_MAP for 'x', unless x_MAP is missing (in which
    case it will use x_final and issue a warning).  The 'rmse' field itself is the
    same as 'x_rmse'. 
    """
    warned_about_missing_x_MAP = False
    make2d = lambda X: X.reshape(X.shape[0], -1)
    for i, datai in enumerate(data):
        log.debug(f"_post_load for {root_folder}:{params_files[i]}.")
        if "x_MAP" not in datai and not warned_about_missing_x_MAP:
            log.warning("Field 'x_MAP' is missing. x_rmse will be computed using x_final.")
            warned_about_missing_x_MAP = True
        if "Y" in datai:
            datai["Y"] = np.squeeze(datai["Y"][-1,:,-1])
        for v in ["x","la","mu"]:
            var_name = v[0].upper() + v[1:]
            if var_name in datai:
                U = datai[var_name] # La, Mu, X
                u_target = datai[f"{v}_final" if (v != "x" or "x_MAP" not in datai) else "x_MAP"].flatten()
                # We reshape the data so that we have time x cells, and we then average across cells
                datai[f"{v}_rmse"]  = np.sqrt(np.mean((make2d(U) - u_target)**2,axis=1))
        if "x_rmse" in datai:
            datai["rmse"]  = datai["x_rmse"]
        datai["file"]  = params_files[i]
        datai["seed"]  = load_params_from_file(root_folder, params_files[i])["seed"]
        datai["params"]= load_default_params().update(load_params_from_file(root_folder, params_files[i]))
    
    return sorted(data, key = lambda d:d["seed"])

class SweepDataset:
    """Class for loading sweeps where we varied one of the model parameters 
    and held the others fixed at their 'center' values.
    """
    
    def __init__(self, root_folder, valid_params = ["be", "ga", "sd", "tau_gc", "tau_mc", "tau_pg"], time_field = "T"):
        self.root_folder = root_folder

        INFO("Loading dataset in %s", root_folder)    
        self.df, self.params = load_params_from_folder(self.root_folder)

        INFO("Loaded data for %d parameters.", len(self.df))
        self.valid_params = valid_params

        self._data = {}
        self._time_field = time_field
        
    def get_valid_params(self):
        return self.valid_params

    def get_loaded_params(self):
        return list(self._data.keys())

    def get_instance(self):
        p = self.get_loaded_params()[0]
        v = self.get_param_vals(p)[0]
        return self._data[p][v]
    
    def get_instance_of_field(self, fld):
        inst = self.get_instance()
        return inst[0][fld]
    
    def get_results_fields(self):
        inst = self.get_instance()
        flds = list(inst[0].keys())
        return flds

    def get_time(self):
        inst = self.get_instance()        
        return inst[0][self._time_field]
    
    def get_num_trials(self):
        inst = get_instance()
        return len(inst)
    
    def __getitem__(self, key):
        """Access items as params, value, field."""
        param, value, fld = key        
        if param not in self._data:
            raise KeyError(f"Parameter {param} not in data. Run 'load_sweep_for_parameter' to load it.")
        results = [trial_data[fld] for trial_data in self._data[param][value]]
        if type(results[0]) is np.ndarray:
            results = np.array(results)

        return results
    
    def get_param_vals(self, which_param):
        """Return a sorted list of the values used for the specified parameter."""
        
        if which_param not in self.valid_params + ["S", "k"]:
            raise KeyError("{} not in valid params: {}".format(which_param, self.valid_params))

        return sorted(list(set(self.df[which_param])))
        
    def get_center_vals(self, k=3, inds = None):
        """Get the center value for the specified parameter.
        Optional parameter 'inds' specifies which parameter 
        this is in the list of values used for each parameter.
        """
        param_vals  = {p:self.get_param_vals(p) if p != "k" else [k] for p in self.valid_params + ["k"]}
        n_vals      = {p:len(pv)        for p, pv in param_vals.items()}
        center_inds = {p:nv//2          for p, nv in n_vals.items()}        
        for p, n in n_vals.items():
            if not (n % 2):
                log.warning("Number of values {} for parameter {} is even, assumed {} is center index.".format(n, p, center_inds[p]))
        return {p:pv[center_inds[p]] for p, pv in param_vals.items()}

    def load_sweep_for_parameters(self, S, param, vars=["X", "La", "Y", "Mu", "x_final", "la_final", "mu_final", "t_final", "T"], load_center_from_Sk = False, n_max = 1000):
        """Load all the sweep data for the specified S and parameter.
        'S' is the number of sisters, 'param' is a string specifying a
        single parameter, a list of strings containing multiple
        parameters, or a dictionary where the keys are parameters and
        values are the specific parameter values to be loaded. The
        result is stored in the _data dictionary and can be accessed
        as [param, val, field].
        
        OPTIONAL ARGUMENTS:
        'vars': Specifies the list of variables to load.
        'load_center_from_Sk': [False] Whether to load the results for the center 
        parameter values from the Sk runs
        'n_max':[1000] Maximum number of results to load.
        
        RETURNS: data
        'data': A dictionary, indexed as [param][value]
        Each item of the dictionary is an array of results, one 
        for each parameter directory found.
        """
        
        vars = list(set(vars) | {"T","Y"}) # Append Y and T the list of variables if not already in it.

        if type(param) is list:
            params_and_vals = {p:self.get_param_vals(p) for p in param}
        elif type(param) is dict:
            params_and_vals = dict(param)
        else:
            params_and_vals = {param:self.get_param_vals(param)}
        
        INFO(f"Loading sweep for S = {S} and parameters {list(params_and_vals.keys())}.")
        self.S = S
        for param, vals in params_and_vals.items():
            INFO(f"Loading sweep for parameter {param} = {vals}.")
    
            # The mask to apply to the dataframe to get the desired values
            base_mask = (self.df["S"] == S)
            ctr  = self.get_center_vals()        
            for p,v in ctr.items():
                if p != param:
                    base_mask &= (self.df[p] == v)
    
            data = {}
            for v in vals:
                if load_center_from_Sk and v == ctr[param]:
                    # Load it from sweep S_k
                    data[v],*_ = load_Sk(S,ctr["k"],vars=vars)
                else:
                    mask = copy(base_mask) & (self.df[param] == v)
                    df_sub = self.df[mask]
                    params_files = list(df_sub["file"])
                    n_params_files = len(params_files)
                    # p[:-5] to strip the .json
                    params_dirs = [p[:-5] for p in params_files if os.path.isdir(os.path.join(self.root_folder, p[:-5]))]
                    n_params_dirs = len(params_dirs);
                    INFO(f"Found {n_params_files} parameter files and {n_params_dirs} corresponding directories for S = {S}, {param} = {v}.")                    
                    if n_params_dirs>0:
                        data[v] = [load_results(self.root_folder, d, vars=vars) for d in params_dirs[:n_max]]
                        data[v] = _post_load(data[v], params_files, self.root_folder)
    
            self._data[param] = data
        return self._data
    
    def apply(self, name, f, flds, data_field = None):
        """Applies the function f to the data in this sweep for all the specified fields
        and adds the result as a property with the specified name to this object.
        
        RETURNS: The computed statistic as a [param,value,fld] dictionary.
        """
        if name in self.__dict__:
            raise KeyError(f"Object already contains a property called '{name}'. Choose another name.")
        if type(flds) is not list:
            flds = [flds]
        stat   = {}
        params = {p:self.get_param_vals(p) for p in self.get_loaded_params()}
        INFO("Computing '{}' for parameters {} and fields {}.".format(name, self.get_loaded_params(), flds))
        for p, pv in params.items():
            INFO(f"Computing '{name}' for '{p}'.")
            for v in pv:
                for fld in flds:
                    argument = self[p,v,fld] if not data_field else self.__dict__[data_field][p,v,fld]
                    stat[p,v,fld] = f(argument) 
        self.__dict__[name] = stat
        INFO(f"Done computing '{name}'.")
        return stat


def full_load(root_folder, params_dirs, params_files, vars, warn_on_missing=True):
    return _post_load([load_results(root_folder, d, vars=vars, warn_on_missing=warn_on_missing) for d in params_dirs], params_files, root_folder)

class FreeSweepDataset:
    """Class for loading sweeps where we've varied parameters freely.
    Data are loaded by specifying the values of each field, in the
    order specified by valid_params. Such a specification must
    determine exactly one parameter file.    
    """
    def select(self, param_vals):
        mask = True
        for p, v in zip(self.params, param_vals):
            mask &= (self.df[p] == v)
        return self.df[mask]
    
    def __init__(self, root_folder, params, time_field = "T", load_on_demand = True):
        self.root_folder = root_folder

        INFO("Loading dataset in %s", root_folder)    
        self.df, self.params_per_file = load_params_from_folder(self.root_folder)

        multivalued_params = [p for p in self.df.columns if len(set(self.df[p]))>1]
        if not all(p in multivalued_params for p in params):
            raise ValueError(f"Not all {params=} were in the set of multivalued_params={multivalued_params}.")
        self.params = params
        self.pind = {p:self.params.index(p) for p in self.params}

        self.param_value_combs_available = list(set([tuple(t) for t in self.df[self.params].values.tolist()]))
        INFO(f"Data available for {len(self.param_value_combs_available)} parameter combinations.")
        self.param_value_combs_loaded = []
        
        # Data will be indexed by tuples of parameter values in the order specified in valid_params        
        self._data = {} 
        self._time_field = time_field

        self.load_on_demand = load_on_demand
        
    def get_params(self):
        return self.params

    def get_param_value_combs_available(self):
        return self.param_value_combs_available

    def get_param_value_combs_loaded(self):
        return list(self._data.keys())

    def get_instance(self):
        p = self.get_loaded_params()[0]
        return self._data[p]
    
    def get_instance_of_field(self, fld):
        inst = self.get_instance()
        return inst[0][fld]
    
    def get_results_fields(self):
        inst = self.get_instance()
        flds = list(inst[0].keys())
        return flds

    def get_time(self):
        inst = self.get_instance()        
        return inst[0][self._time_field]
    
    def get_num_trials(self):
        inst = get_instance()
        return len(inst)

    def _load_item(self, key):
        if callable(self._data[key]):
            self._data[key] = self._data[key]()
        return self._data[key]
    
    def __getitem__(self, key):
        """Access items either as (param_combs) or ((params_combs), fld)"""
        if type(key) is tuple and type(key[0]) is not tuple:
            comb = key
            fld  = "all"
        elif type(key) is tuple and type(key[0]) is tuple and type(key[1]) is str:
            comb = key[0]
            fld  = key[1]
        else:
            raise KeyError(f"Don't know what to do for key '{key}'.")

        if comb not in self._data:
            raise KeyError(f"Parameter combination {comb} not in data. Run 'load_sweep' to load it.")

        results = self._load_item(comb) if fld == "all" else [self._load_item(trial_data[fld]) for trial_data in self._data[comb]]
        if type(results[0]) is np.ndarray:
            results = np.array(results)

        return results
    
    def get_param_vals(self, which_param):
        """Return a sorted list of the values used for the specified parameter."""
        
        if which_param not in self.params:
            raise KeyError(f"{which_param} not in valid params: {self.params}")

        ind_param = self.params.index(which_param)
        return sorted(list(set(self.combs[ind_param])))

    def clear_data(self):
        INFO(f"Clearing data from {len(self._data)} loaded entries.")
        self._data = {}
        
    def load_sweep(self, param_combs = "all", vars=["X", "La", "Y", "Mu", "x_final", "la_final", "mu_final", "t_final", "T", "x_MAP", "x_input"], n_max = 1000, warn_on_missing=True):
        """Load the sweep data. If param_combs is "all", all the
        available combinations will be loaded. If it's a tuple, it's
        interpreted as parameter combination and the data for that
        combination will be returned. If param_combs is a list, the
        data for each combination in the list is returned in a
        dictionary indexed by the combinations. As a dictionary it
        specifies the subset of parameters we want loaded. 
        
        OPTIONAL ARGUMENTS:
        'vars': Specifies the list of variables to load.
        'n_max':[1000] Maximum number of results to load.
        
        RETURNS: data
        'data': A dictionary, indexed by combination.
        Each item of the dictionary is an array of results, one 
        for each parameter directory found.
        """
        
        vars = list(set(vars) | {"T","Y"}) # Append Y and T the list of variables if not already in it.

        return_mode = "multiple"
        if param_combs == "all":
            param_combs = self.param_value_combs_available
        elif type(param_combs) is tuple:
            if param_combs not in self.params:
                raise ValueError("No data available for {self.params} = {param_combs}.")
            param_combs = [param_combs]
            return_mode = "singleton"
        elif type(param_combs) is list:
            if not all([comb in self.param_value_combs_available for comb in param_combs]):
                raise ValueError("Not all desired parameter value combinations were available.")
        elif type(param_combs) is dict:
            keep_combs = []
            for comb in self.param_value_combs_available:
                keep = True
                # Go through all the desired params and their allowed values
                # and make sure this combination has one of each
                for param,vals in param_combs.items():
                    if comb[self.pind[param]] not in vals:
                        keep = False
                        break
                if keep:
                    keep_combs.append(comb)
            INFO(f"Subsetting to {len(keep_combs)} of {len(self.param_value_combs_available)} parameter combinations.")
            param_combs = keep_combs
        else:
            raise ValueError("Don't know what to do for {param_combs=}.")
        
        INFO(f"Loading data for {len(param_combs)} of {len(self.param_value_combs_available)} available parameter combinations.")
        INFO(f"{len(self._data)} records present before load.")
        data = {} 
        for comb in param_combs:
            df_sub = self.select(comb)
            params_files = list(df_sub["file"])
            n_params_files = len(params_files)
            # p[:-5] to strip the .json
            params_dirs = [p[:-5] for p in params_files if os.path.isdir(os.path.join(self.root_folder, p[:-5]))]
            n_params_dirs = len(params_dirs);
            INFO(f"Found {n_params_files} parameter files and {n_params_dirs} corresponding directories for {self.params} = {comb}.")                    
            if n_params_dirs>0:
                # A function to load the data when needed
                #data[comb] = partial(_post_load, [load_results(self.root_folder, d, vars=vars) for d in params_dirs[:n_max]], params_files, self.root_folder)
                data[comb] = partial(full_load, self.root_folder, params_dirs[:n_max], params_files, vars, warn_on_missing)
                if not self.load_on_demand: # If we want th data now, load it right away
                    data[comb] = data[comb]()
                    
        self._data.update(data)
        INFO(f"Done loading. {len(self._data)} records present.")        
        return self._data
        
        
def describe(obj, depth = 0, full = True, prefix = ""):
    """Gives a type description of the specified object.
    
    OPTIONAL ARGUMENTS:
    'depth': [0] The maximum depth to dig down into.
    'full': [True] When arriving at a level where none of
    of the objects are dicts or lists, whether to still
    iterate through and describe the types of the objects.
    'prefix': "" A prefix to apply to each description line.
    """
    dig_types = [list, dict]
    spcr = " " * depth
    if type(obj) == list:
        print("{}LIST of {} items of type {}".format(spcr, len(obj), type(obj[0])))
        describe(obj[0], depth = depth + 1, full = full)
    elif type(obj) == dict:
        keys = list(obj.keys())
        print("{}DICT with keys {}".format(spcr, list(keys)))        
        if full is True:
            if not any([type(obj[k]) in [dict, list] for k in keys[1:]]):
                [describe(obj[k], depth=depth+1, prefix = "{:>8s}: ".format(str(k))) for k in keys]
            else:
                describe(obj[keys[0]], depth = depth + 1, full = full)
        else:
            describe(obj[keys[0]], depth = depth + 1, full = full)
                    
    elif type(obj) == np.ndarray:
        print("{}{}NDARRAY with shape {}".format(spcr, prefix, obj.shape))
    else:
        print("{}{}ITEM of type {}".format(spcr, prefix, type(obj)))


def find_spectrum_peaks(A, fr, rng = [1,300], min_freq = 40, method = "find_peaks", **kwargs):
    """Returns a list of the peaks in the spectrum provided.
    
    ARGUMENTS:
    'A': Amplitude spectra provided as a 2D,  freqs x objects nd-array.
    'fr': The frequencies corresponding to the rows of 'A'.
    
    OPTIONAL:
    rng: [1,300] The frequency range to find the peaks in.
    min_freq: [40] The minimum frequency for a peak.
    **kwargs: Passed on to find_peaks
    
    RETURNS: (fpks, vals)
    'fpks': A list containing the list of peaks for each column in 'A'.
    'vals': The corresponding peak values.
    """

    if len(A.shape)==1:
        A = np.reshape(A, (A.shape[0],1))
    n_cols = A.shape[1]
    
    freqs = fr[(fr>rng[0]) & (fr<rng[1])]
    amps  =  A[(fr>rng[0]) & (fr<rng[1])]

    method = method.lower()
    if method == "find_peaks":
        peak_finder = lambda a: find_peaks(a, **kwargs)[0]
    elif method == "argrelextrema":
        peak_finder = lambda a: argrelextrema(a, comparator = np.greater, **kwargs)[0]
    elif method == "find_peaks_cwt":
        peak_finder = lambda a: find_peaks_cwt(a, **kwargs)
    else:
        raise ValueError(f"method must be one of 'find_peaks' or 'argrelextrema', was '{method}'.")
            
    all_peaks_per_column   = [peak_finder(amps[:,i]) for i in range(n_cols)]

    valid_peaks_per_column = [ipks[freqs[ipks]>=min_freq] for ipks in all_peaks_per_column]
    
    peak_amps  = [amps[ipks, i] for i, ipks in enumerate(valid_peaks_per_column)]
    peak_freqs = [freqs[ipks]   for ipks in valid_peaks_per_column]
    
    return peak_freqs, peak_amps

def create_and_run_olfactory_bulb(p, keep_every = 100, report_frequency = 10, **kwargs):
    """ Creates and runs an instance of an olfactory_bulb using the
    parameters in the dictionary p. The default parameters are first
    loaded and then updated with those in p, so p doesn't need to
    contain all fields.  The input odour used is the default odour for
    the given sparsity level if x is None.  x can be set to a single
    odour of length N to run a user defined odour.  It can also be an
    array of user defiend odours of length N.

    If only one odour was run, the results themselves will be
    returned. If more than one odour was run, an array containing the
    results for each odour will be returned.
    """

    params = load_default_params()
    params.update(p)
    p  = params


    if "which_odour" not in p or p["which_odour"] == -1:
        x  = get_x_true(p["N"], p["k"], spread = p["spread"])
    else:
        # Generate random inputs
        np.random.seed(p["which_odour"])
        INFO("Using random seed {} to generate an odour with {} random components set to 1.".format(p["which_odour"], p["k"]))
        x = np.random.permutation(p["N"]) < p["k"]
        x = x.astype(float)

    ind_nz = np.where(abs(x)>1e-6)[0]
    INFO(f"Odour has {len(ind_nz)} indices with absolute value > 1e-6.")
    INFO( "First few as (index, value) pairs: {}".format([(ind, x[ind]) for ind in ind_nz[:min(len(ind_nz),5)]]))

    OB   = OlfactoryBulb(**p)    
    odor = Odor(x, OB.A, amp = p["amp"], plume=p["plume"], t_on = p["t_on"], t_off=p["t_off"])

    results = OB.run_sister(odor, p["t_end"], p["dt"], keep_till = p["keep_till"], keep_every = keep_every, report_frequency = report_frequency, **kwargs)
    results["x_input"] = np.copy(x)
    
    return results
