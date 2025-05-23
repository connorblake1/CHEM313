# 2D Execution and Markov State Model
## File Descriptions
The following files have been entirely rewritten to fit my needs or added to the repository.
- `general.py` contains the main driver algorithm
- `sampling.py`, `training.py` contain subroutines called in `general.py` for the loss and Langevin dynamics
- `config.py` contains a list of custom potentials whose parameters can easily be tweaked. 
- `global_utils.py` contains the hyperparameter `max_K` which dictates how many metastable basins to look for
- `fem_utils.py` computes the "exact" numerical committor using finite element method on the 2D domain. It must be run in a conda environment containing the library fenics which is most easily installed via conda (or try `fenics.yml`). This environment does not play nicely with pytorch. See below for more details.


## Running Simulations
1. Change the following items:
    - `config.py`: set up the potential by changing stuff in the block \<CHANGE THESE PARAMETERS>
    - `general.py`:  set up the sampling run by changing the `centers_k` tensor in the \<CHANGE THESE PARAMETERS> and the `cmask`. `centers_k` controls where the trajectories are launched from (up to $K$). `cmask` is a bitmask dictates how many outputs of the neural net are active at at time. If you only want $k < K$ predictions, then `cmask` will have $k$ 1s and `centers_k` only needs to have $k$ centers.
    - `global_utils.py`: `max_K` must be changed
2. Run `python general.py` in the conda environment `md_sims` from the directory `2D`.

## Computing Markov State Models
See `msm.py`

## Running FEM
1. Change to the `fenics` conda environment.
2. Set up your potential in `config.py` as above.
3. `python config.py`

For another example, run `python fem_mb_exec.py` to see the Mueller Brown potential (compared to the exact committor they put in the repo)