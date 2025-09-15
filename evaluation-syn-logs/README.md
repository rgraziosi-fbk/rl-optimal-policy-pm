# Synthetic log generation and simulation evaluation 
 
## Installation

1. Create a new conda environment with Python 3.10: `conda create --name syn_evaluation python=3.10`
2. Inside the newly created environment, install required packages: `pip install --no-cache-dir -r requirements.txt`

## Running the script

Once installed all the packages, you can execute the tool from a terminal specifying the following parameters:

* `-t`: to specify if you want to generate a synthetic log or to produce a simulation evaluation.  Available options: synthetic, cluster e cql.
* `-o (optional, default=simulation)`: Refers to the name for the simulated event_log file.
* `-n (optional, default=1000)`: Total number of traces to generate.
* `-r (optional, default=False)`: With value True if you want to generate a rare synthetic log.
* `-m`: To generate a simulation evaluation, the path of folder to find all the cluster mdp models or CQL models components 
* `-p`: Only for cluster mdp models to specify the policy to be used for simulation evaluation

**Example of basic execution for synthetic logs:**
To generate a synthetic event log of 100 traces

- standard synthetic model
```shell
python main.py -t synthetic -o log_100 -n 100
```

- *rare* synthetic model 
```shell
python main.py -t synthetic -o log_rare_100 -n 100 -r True
```

Logs are saved in the folder `output` 

**Simulation evaluation:**

- ***MDP policies:***

The policies are contained in `clusters_model`, each subfolder corresponds to one event log and contains the two pickle files `_model.pkl` and `_pickle.pkl` defining the clusterization model and all the policies learned for every choice of the scale factor *h*.

To generate a simulation evaluation for one of the MDP policies (e.g. the *expQ2*) learned for one of the synthetic log (e.g. the *log_2000*) run the following command:

```shell
python main.py -t cluster -o sim_eval_log_2000_cluster_100_expQ2 -n 2000 -p expQ2 -m cluster_models/log_2000_100/
```


- ***CQL policies:***

The policies are contained in `cql_recommender/output_data`, each subfolder corresponds to one event log and a certain training epoch (50th or 100th) and contains a configuration file `config.pkl` and the model weights `.d3`.

To generate a simulation evaluation for one of the CQL policies (e.g. the one at *epoch 50*) learned for one of the synthetic log (e.g. the *log_2000*) run the following command:
```shell
python main.py -t cql -o sim_eval_log_2000_cql_50 -n 2000 -m cql_recommender/output_data/log_2000_e50
```

Logs and reward files are saved in the folder `output`


**Analysis of the reward obtained with the simulation evaluation:**

The following scripts are used to aggregate the simulation rewards, compare the performance of different policies on the same event log, and perform statistical significance tests:

- `output_analysis.py`: performs aggregated analysis on evaluation simulations.
- `output_analysis_evolution.py`: performs aggregated analysis on evaluation simulations, considering policies activated at different prefix lengths.
- `p-test_calculator.py`: performs pairwise statistical significance tests between policies and computes the corresponding p-values.
- `p-test_calculator_evolution.py`: performs pairwise statistical significance tests between policies and computes the corresponding p-values, considering activation at different prefix lengths.
