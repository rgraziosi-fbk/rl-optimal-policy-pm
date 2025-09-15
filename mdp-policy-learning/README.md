# Policies Learned via Trace Clustering, MDP Construction, and Dynamic Programming 
 
## Installation

1. Create a new conda environment with Python 3.10: `conda create --name mdp_policy python=3.10`
2. Inside the newly created environment, install required packages: `pip install --no-cache-dir -r requirements.txt`

## Scripts description:

The relevant scripts for this project are contained in the folder `src`.

**Training MDP policy pipeline:**

`Training_mdp_policy.py`: contains the full pipeline to learn a policy starting from the event log using clustering, MDP construction, and Dynamic Programming.

It takes as input variables the working folder name `folder_name`, the event log name `file` and the scale factor *h* to use in the Dynamic Programming algorithm `scale_factor`.

The construction of the MDP employs the clustering of traces using the best cluster number `n_clusters` as derived from the wss (Within-Cluster-Sum of Squared) analysis.

The final outputs are:
- The policy file saved in `cluster_data/[folder_name]/output_policies` with naming convention: `[file]_[n_clusters]_training_policy_opt_DP_[scale_factor].csv`
- Two pickles files used to integrate the policy in the simulator models for evaluation, they are saved in `cluster_data/[folder_name]/output_logs` with naming conventions: `[file]_[n_clusters]_training_model.pkl` and `[file]_[n_clusters]_training_pickle.pkl`


(*On Windows*: if you encounter the error
`ImportError: cannot import name 'Digraph' from 'graphviz.dot'` change the import in the `pm4py` module `pm4py\visualization\common\gview.py` into `from graphviz import Digraph`.)


**Utility scripts:**

- `Clusterer_Silhouette_analysis.py`: performs the wss (Within-Cluster-Sum of Squared) analysis to indentify the best cluster number for each event log.
- `Preprocessing_logs.py`: performs some preprocessing manipulation on event logs: computes event durations, defines rewards and splits real logs into training and test sets.


