# Supplementary Notes
Connor Blake, 5/23/25

Github Repo: [Final Project Repository](https://github.com/connorblake1/CHEM313/tree/master/final_project)
    - New files I wrote in `sc_committor/2D`: `general.py`, `utils.py`, `config.py`, `global_utils.py`, `fem_utils.py`, `msm.ipynb`

## Experiments I Left Out of the Presentation:
    - I implemented FEM solver tools `fem_utils.py` into my workflow for validating models. These can be found in the files `run_data/*_grid_comparison.pdf`. These serve as "exact" model validations. Note that q is plotted (not p), so sometimes the plots will look like there is very high error when in fact it's highly accurate.
    - See `msm.py` for analysis of the solved MFPT for the linear 5 well potential shown in the presentation. By changing the first cell under "Markov State Model", different geometries can be analyzed.
    - For other geometries (not just the linear 5), see e.g. `wells_dist_square_0_a0_b3_K4_all.pdf` vs `wells_dist_square_0_a0_b3_K2_all.pdf` to compare the instability of the original model + my modifications. This model has multiple valid pathways.

## ML Contributions:
    - I refactored the entire simulation suite for n-dim potentials ($V$ defined explicitly, not via an MD package) to handle the $K$ basins instead of 2 basins. While this seems trivial, making sure this was working properly across all files and subroutines took me a significant amount of time (even with AI + decent coding experience, 6+ hours). This included more than just packing all the separate tensors into tensors with a new dimension (which by itself required rewriting every line) - it also included handling various cross-transit logic and other information that is exported to the Markov State Model. For changes, see all the new code written in `general.py`
    - The NN architecture was modified to output $q_k = 1 - p_k$. By enforcing a softargmax on the output of the neural network, it is guaranteed to generate a valid forward committor distribution.
    - By adding a $\mathcal{O}(N \log N)$ clustering method to the NN outputs, significant information on intermediates can be extracted from even very poor committor predictions.
    - I restructured the notation (ie leaving vs entering committors, etc) to be 1:1 compatible with a continuous-time Markov chain so that it can be easily computed with standard techniques. See `msm.py`

## Key Takeaways:
    - The main drawback of the log-committor method is the ability to handle metastable intermediates, as noted in the paper. They specifically suggest some kind of hybrid Markov State Model approach to fix this, which I have done.
    - Even if a NN committor cannot estimate a rate properly with metastable intermediates, a simple clustering technique can still make its results useful.
    - With an increase linear in $K$ in the training time and space complexity, reaction rates can be approximated using the log-committor method that cannot be approximated with the base model.
    - The original authors observe significant instability in reaction rates when the potential had metastable intermediates. Some plots (eg 5b) plot trendlines that look almost meaningless as the prediction jumps orders of magnitude up and down around the actual value. This looks ok on a log-scale, but these are extremely inaccurate which is mostly glossed over.

## Some Defenses of Potential Critiques:
    - No MD simulations?: PhD students in my group advised me not to use my PI's Midway credits. Because my model focused on metastable intermediates, I would have ideally used Aib9 to test my model as dialanine peptide doesn't have significant metastable intermediates. The simulation time they used for AIB9 was 96 hours on a single GPU, and I could not tie up my laptop's (smaller) GPU for 4 days + all the verification time.
    - Two dimensions?!: I wanted to be able to verify, with FEM the "exact" committor. Most python FEM packages can only handle 2 and 3D, and the meshing scales exponentially in the number of dimensions. In order to retain accurate models, I had to limit these to rather small grids and in 2D. Also, because NNs learn arbitrary maps and have randomized weights, it seems plausible to me that a complicated 2D model has the same complexity as e.g. 5D with the same number of basins.

## Question for you:
Similar to the question I asked after my presentation, I don't have a good sense of what is considered an open problem in this area of modeling. I.e. what is trivial vs hard vs impossible vs imprecise vs too theoretical etc. If I am interested in this field of research, is there a good review article that summarizes open/important challenges?
