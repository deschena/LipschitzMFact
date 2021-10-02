# Optimization for Machine Learning Project
Justin Deschenaux, Guillaume Follonier and Jun Han

## Goal
Check whether the convergence analysis of SGD on Lipschitz continuous and convex function on convex domain can guide the choice of the stepsize in training of matrix factorization models (non-convex), and whether it is better than performing a grid search on the hyperparameter space.


## Files
- `data` this folder contains the MovieLens 100k dataset that we used for our experiments.
- `MFact.py` contains the code for the matrix factorization model, including initializations.
- `helpers.py` contains the code for preprocessing, computing errors and other statistics about the data.
- `run.py` contains the code to perform our main experiment.
- `training_loops.py` contains many different methods to train the matrix factorization model.
