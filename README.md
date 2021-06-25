# Olfactory bulb model with sister mitral cells
- Code to reproduce the simulations and figures in Tootoonian et al. 2021.
## Requirements
- Code was written and tested using Python 3.8.5 provided by Anaconda.
- Other versions/distributions of Python 3.8+ should also work with minimal modification.
- This code requires [cvxpy](https://www.cvxpy.org/install/). This can be installed using `pip install cvxpy`.
## Basic setup
- Download this repository to a directory of your choice, henceforth `sister-mcs-release`.
- To test the code, navigate to `sister-mc-release/code/test` 
- Run `./run.sh`.
- This will run a basic instance of the model.
- Once the run is complete, process the results using `python proc.py`
- This will create `proc.pdf`, showing some of the outputs.
## Code structure
- Most of the code is in the `code` subfolder.
- The file containing the olfactory bulb model is [olfactory_bulb.py](code/olfactory_bulb.py).
- An example of basic usage of the model can be found in the `create_and_run_olfactory_bulb` in [datatools.py](code/datatools.py).



