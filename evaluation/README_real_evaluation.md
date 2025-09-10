# Evaluation Method

### Preprocessing for the definition of the Simulation model

- Define the DFG.json file starting from the BPMN file defining the process model with *preprocessing_bpmn.py*
- Define the decision points for each gateway in the BPMN model
- Define input_NAME_EXPERIMENT.json where specify:
  - "start_timestamp": date_start of the simulation
  - "interTriggerTimer": arrival rate of the traces in the simulation
  - "resource": Role with its resources and associated calendar
  - "activity_to_role": for each activity the Role that can perform it
  - "processing_time": for each activity the distribution function
  - "resource_to_role": for each resource the corresponding role
  - "TRACE_ATTRIBUTES"
  - "EVENT_ATTRIBUTES"
  - "path_decision_tree": path where the files “NAME_EXPERIMENT_decision_points.json” and pkl are located
  - "gateway_probability": probability associated to each gateway and the related flows from it
  - "ENV_activities": list of activities charged to the environment
  - "AGENT_activities": list of activities charged to the agent
  - "AND_node_termination": id name of the join parallel gateway (key of the dictionary is the id of gatewayDirection="Diverging" and the value is the is of gatewayDirection="Converging" from BPMN file)

See the files contained in *input* folder for BPIC12 and BPIC17 real logs or for more details in the parameters see [Rims Tool Documentation](https://francescameneghello.github.io/RIMS_tool/index.html)

### After the preprocessing step run the simulations with the file *main.py* by defining the arguments of the *main_explicit_arguments()*