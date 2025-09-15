import os
import time
import pandas as pd
import pm4py
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.objects.log.obj import Event
from sklearn.preprocessing import MinMaxScaler



def create_mdp(path, tracefilter_log_pos, reward_label="cluster:reward", cumulative_reward=True, scale_reward=True):
	OUTPUT_PATH = path.replace("logs", "mdps").replace('.xes', '.csv')
	cols_headers = ["s","a","s'","p_r","reward","number_occurrences"]
	reward_dict = dict()

	# create dictionary with reward and count
	for trace in tracefilter_log_pos:
		event_name = getEventResourceName_cluster(trace[0], trace)
		for event in trace[1:]:
			reward = float(event[reward_label]) if reward_label in event.keys() else 0
			new_event_name = getEventResourceName_cluster(event, trace)
			s1, s2, a = get_states_action_cluster(event_name, new_event_name)
			key = (s1, a, s2)
			if key not in reward_dict.keys():
				reward_dict[key] = {'count': 0, 'sum': 0.0, 'avg_rew': 0}
			reward_dict[key]["sum"] += reward
			reward_dict[key]["count"] += 1
			event_name = new_event_name

	# compute average reward for every transitions
	max_occurrences = 0
	for (s1, a, s2), value_dict in reward_dict.items():
		if value_dict["count"] > max_occurrences:
			max_occurrences = value_dict["count"]
		value_dict['avg_rew'] = value_dict['sum'] / value_dict['count'] if value_dict['count'] > 0 else 0
		value_dict['reward'] = value_dict['avg_rew'] if ("END" in s2 or not cumulative_reward) else 0

		# create MDP dataframe
		# transform reward_dict into df
		tmp_df = pd.DataFrame.from_dict(reward_dict, orient='index')
		# rename index labels and transform to columns
		tmp_df.index.set_names(["s", "a", "s'"], inplace=True)
		tmp_df.reset_index(inplace=True)
		# create state_action key
		tmp_df["s_a"] = tmp_df["s"] + "_^_" + tmp_df["a"]
		# compute state_action count
		group_df = tmp_df[['s_a', 'count']].groupby('s_a').sum()
		state_action_count_dict = {k: v['count'] for k, v in group_df.to_dict('index').items()}
		tmp_df["s_a_count"] = tmp_df["s_a"].map(state_action_count_dict)
		# define probability transitions
		tmp_df["p_r"] = tmp_df["count"] / tmp_df["s_a_count"]
		# define MDP, columns = [s, a, s', p_r, reward, number_occurrences]
		mdp_df = tmp_df[["s", "a", "s'", "p_r", "reward", "count"]].rename(columns={"count": "number_occurrences"})
		# define and insert first row
		first_row = pd.Series({"s": "<>", "a": "START", "s'": "<START>",
							   "p_r": 1.0, "reward": 0.0, "number_occurrences": max(tmp_df["s_a_count"])})
		mdp_df = pd.concat([first_row.to_frame().T, mdp_df], ignore_index=True)


	# optional rescaling of reward and number_occurrences
	if scale_reward:
		mmrew = MinMaxScaler().fit(mdp_df[["reward"]])
		mdp_df["reward"] = mdp_df["reward"].map(lambda x: 0 if x == 0 else mmrew.transform([[x]])[0][0])

	# sort and export mdp in csv
	mdp_df = mdp_df.sort_values(by=["s", "a", "s'"])
	mdp_df.to_csv(OUTPUT_PATH, index=False)
	return OUTPUT_PATH

def get_states_action_cluster(event1, event2):
	s1 = "<" + event1.split("AZIONE")[-1].replace('<', '')
	parts = event2.split("AZIONE")
	s2 = "<" + parts[-1].replace('<', '')
	a = parts[0].split(' | ')[0].replace("<", "").replace(">", "").rstrip()
	return s1, s2, a

def getEventResourceName_cluster(event, trace):
	if event["concept:name"] in 'START':
		return "<" + event["concept:name"] + ">"
	else:
		try:
			return "<" + event["concept:name"] + " | " + event["cluster:prefix"] + ">"
		except:
			print(event)
			print(trace.attributes["concept:name"] + '\n\n')


def preprocessBPI(path, keep_1event_loops=True):
	# This method processed the clustered event log take care of environment activities in order to construct the mdp
	file_name_suffix = '_preprocessed_wloops.xes' if keep_1event_loops else '_preprocessed.xes'
	output_path = path.replace('.xes', file_name_suffix)
	log = pm4py.read_xes(path)
	begin_event = Event()
	begin_event["concept:name"] = ""

	# definition of environment activities
	env_activities_BPI12 = ["O_ACCEPTED", "A_CANCELLED", "O_SENT_BACK", "O_DECLINED"]  # holds for BPIC2012 and all the synthetic logs
	env_activities_BPI17 = ["O_Accepted", "A_Cancelled", "O_Returned", "O_Refused"]  # defined in the bpic_student report as purely client activities

	for trace in log:
		last_event = Event()
		last_event["concept:name"] = ""
		for event in trace:
			# environment activities enter in the state definition but cannot be used as actions
			if event["concept:name"] in env_activities_BPI12 + env_activities_BPI17:
				# select id of previous event
				idx = [x for x, e in enumerate(trace) if e == event][0] - 1
				previous_event_name = trace[idx]["concept:name"]
				if "AZIONE" in previous_event_name:
					previous_event_name = previous_event_name.split("AZIONE")[0]  # manage consecutive environment events
				event["concept:name"] = previous_event_name + "AZIONE" + event["concept:name"]
				trace[idx]["concept:name"] = "TO_REMOVE"
			# Below repetition of events are compressed to a single evnt, and therefore a simpler mdp is obtained.
			# However, this is not used in the paper
			if keep_1event_loops == False and last_event["concept:name"] == event["concept:name"] and event["concept:name"] != "TO_REMOVE":
				if "duration" in last_event.keys() and "duration" in event.keys():
					event["duration"] = event["duration"] + last_event["duration"]
				idx = [x for x, e in enumerate(trace) if e == last_event][0]
				trace[idx]["concept:name"] = "TO_REMOVE"
			last_event = event

	# filter REMOVED activities
	tracefilter_log_pos = attributes_filter.apply_events(log, ["TO_REMOVE"],
														 parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	# export preprocessed clustered event log
	xes_exporter.apply(tracefilter_log_pos, output_path)

	print("end preprocessing")
	return output_path, tracefilter_log_pos

def main():
	# script used to build all the mad for all the logs considered
	folder_name = "folder1"

	mode = 'training'

	training_file_list = [('BPI_2012_log_eng_clean', '140'),
						  ('BPI_2017_log_strip_newcredit', '100'),
						  ('event_log_2000', '100'),
						  ('event_log_4000', '140'),
						  ('event_log_8000', '120'),
						  ('event_log_16000', '120'),
						  ('event_log_rare_2000', '120'),
						  ('event_log_rare_4000', '120'),
						  ('event_log_rare_8000', '100'),
						  ('event_log_rare_16000', '100')
						  ]
	test_file_list = [('BPI_2012_log_eng_clean', '100'),
					  ('BPI_2017_log_strip_newcredit', '100'),
					  ('event_log_2000', '100'),
					  ('event_log_rare_2000', '120'),
					  ]

	if mode == 'training':
		for file_name, cluster_number in training_file_list:
			mdp_creator_main(folder_name, mode, file_name, cluster_number)
	# this is the new part that generates a MDP for testing with simulations
	# this test MDP is completely independet on the training MDP, this should give a more general MDP that can be use for all the policy (also with different cluster numbers), and different techniques (also DRL)
	elif mode == 'testing':
		for file_name, cluster_number in test_file_list:
			mdp_creator_main(folder_name, mode, file_name, cluster_number)



def mdp_creator_main(folder_name, mode, file_name, cluster_number):
	print("mdp creation")
	cluster_number = str(cluster_number)
	# creates the mdp to learn the policy
	ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
	if mode == 'training':
		type = '_training.xes'
		print('file: ' + file_name + "_" + cluster_number + type)
		path = os.path.join(ROOT, "cluster_data", folder_name, "output_logs", file_name + "_" + cluster_number + type)
		output_path, output_log = preprocessBPI(path, keep_1event_loops=True)
		create_mdp(output_path, output_log, reward_label="cluster:reward", cumulative_reward=True)
	# this generates a MDP for testing
	elif mode == 'testing':
		type = '_ForTEST.xes'
		print('file: ' + file_name + "_" + cluster_number + type)
		path = os.path.join(ROOT, "cluster_data", folder_name, "output_logs", file_name + "_" + cluster_number + type)
		output_path, output_log = preprocessBPI(path, keep_1event_loops=True)
		# uses single rewards at each step and no the cumulative reward
		# IMPORTANT NOTE: even if the mdp use unscaled reward the clusters have been computed using the normalized reward
		create_mdp(output_path, output_log, reward_label="kpi:reward", cumulative_reward=False, scale_reward=False)
	else:
		raise Exception("Error: mode not correct!")

if __name__ == "__main__":
	t1 = time.time()
	print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t1)))

	main()

	t2 = time.time()
	print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t2)))
	print(t2 - t1)