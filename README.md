# Truly batch Model-free Inverse Reinforcement Learning about Multiple Intentions
Code of Truly Batch Model-Free Inverse Reinforcement Learning about Multiple Intentions
This repository contains the implementation of the SIGMA-GIRL algorithm and experiments in simulated domains, together with implementation of algorithms we used to confront our results.

## Available algorithms
- [SIGMA-GIRL](algorithms/pgirl.py)
- [RE-IRL](algorithms/REIRL.py)
- [ML-IRL](algorithms/mlirl.py)
- [CSI](algorithms/CSI.py)
- [SCIRL](algorithms/CSI.py)

## Reproducibility 
To reproduce our results run the following scripts:
- `run_lqg_single_irl` : Run IRL experiments in the lqg environment. It generates the expert trajectories and then performs IRL with all the implemented IRL algorithms.
- `run_gridworld_single_irl` : Run IRL experiments in the gridworld environment. It trains policies in the environment using GPOMDP, then uses them to generate expert trajectories and then performs IRL with all the implemented IRL algorithms.
- `run_clustering_all_gridworld` : Run MI-IRL experiments in the gridworld environment. It loads pregenerated trajectories from multiple experts (`run_gridwolrd_multipleintent.py` can be used to generate these trajectories), and performs clustering using SIGMA-GIRL
- `run_mlirl_clustering` : Run MI-IRL experiments in the gridworld environment. It loads pregenerated trajectories from multiple experts (`run_gridwolrd_multipleintent.py` can be used to generate these trajectories), and performs clustering using ML-IRL
- `run_clustering_all_twitter` : Run MI-IRL experiments using the twitter dataset. It loads the dataset and performs clustering using SIGMA-GIRL
