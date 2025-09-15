import os
from csv import DictWriter
from statistics import mean, stdev, median

import numpy as np
import pandas as pd
import pm4py
import random
import datetime
import time

from pm4py import filter_trace_attribute
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes import exporter as xes_exporter
from pm4py.algo.filtering.log.attributes import attributes_filter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery


AMOUNT_LABEL = "amount"


def splitLog(log_path, percentage):
	tracefilter_log_pos = pm4py.read_xes(log_path)
	output_training = log_path.replace(".xes", "_training.xes")
	output_testing = log_path.replace(".xes", "_ForTEST.xes")

	traces_list = [x for x in tracefilter_log_pos]
	train_l = int(len(traces_list) / 100 * percentage)
	train_log = EventLog()
	test_log = EventLog()
	train_acc_cases_count = 0
	train_decl_cases_count = 0
	train_others_cases_count = 0
	test_acc_cases_count = 0
	test_decl_cases_count = 0
	test_others_cases_count = 0

	for i, t in enumerate(traces_list):
		events = [e["concept:name"] for e in t]
		if i < train_l:
			train_log.append(t)
			if "O_ACCEPTED" in events:
				train_acc_cases_count += 1
			elif "O_DECLINED" in events:
				train_decl_cases_count += 1
			else:
				train_others_cases_count += 1
		else:
			test_log.append(t)
			if "O_ACCEPTED" in events:
				test_acc_cases_count += 1
			elif "O_DECLINED" in events:
				test_decl_cases_count += 1
			else:
				test_others_cases_count += 1

	xes_exporter.apply(train_log, output_training)
	xes_exporter.apply(test_log, output_testing)

	print("Train log:\nTotal cases count: {}\nAccepted cases count: {}\n Declined cases count: {}\n Other cases count:{}".format(len(train_log),
																																 train_acc_cases_count,
																																 train_decl_cases_count,
																																 train_others_cases_count))
	print("Test log:\nTotal cases count: {}\nAccepted cases count: {}\n Declined cases count: {}\n Other cases count:{}".format(len(test_log),
																																 test_acc_cases_count,
																																 test_decl_cases_count,
																																 test_others_cases_count))


def split_lifecycle(lifecycle):
	lifecycle_list = []
	starting_index = 0
	for index, event in enumerate(lifecycle):
		if event["lifecycle:transition"].lower() in ('complete', 'ate_abort') and index + 1 != len(lifecycle): # 'complete'
			lifecycle_list.append(lifecycle[starting_index:index + 1])
			starting_index = index + 1
		elif index + 1 == len(lifecycle):
			lifecycle_list.append(lifecycle[starting_index:])
	return lifecycle_list

def compute_lifecycle_duration(lifecycle):
	last_event = Event()
	last_event["concept:name"] = ""
	timestamp_list = []
	for index, event in enumerate(lifecycle):
		if index == 0:
			start_event = event
			last_event = event
			timestamp_list += [event["time:timestamp"]]
		elif last_event["lifecycle:transition"].lower() in ('start', 'resume') and event["lifecycle:transition"].lower() in ('suspend', 'complete'):
			last_event = event
			timestamp_list += [event["time:timestamp"]]
		elif last_event["lifecycle:transition"].lower() in ('suspend') and event["lifecycle:transition"].lower() in ('resume', 'start', ):  # actually it should never be 'start'
			last_event = event
			timestamp_list += [event["time:timestamp"]]
	# the list should have an even number of elements
	if len(timestamp_list) % 2 != 0:
		Exception("Problem with compute_lifecycle_duration: there is an odd number of timestamps")
	else:  # compute duration of the lifecycle
		duration = 0
		for i in range(int(len(timestamp_list)/2)):
			duration += (timestamp_list[2*i+1] - timestamp_list[2*i]).total_seconds()
	# add the lifecycle to start events
		start_event["duration"] = round(duration, 2)
	# remove all the other events
	for index, event in enumerate(lifecycle):
		if index != 0:
			event["concept:name"] = "TO_REMOVE"


def trace_add_start_end(trace: Trace):
	start = Event()
	start["concept:name"] = "START"
	start["task"] = "START"
	start["time:timestamp"] = trace[0]["time:timestamp"]
	trace.insert(0, start)
	end = Event()
	end["concept:name"] = "END"
	end["task"] = "END"
	end["time:timestamp"] = trace[-1]["time:timestamp"]
	trace.append(end)
	return trace


def add_Event_Duration(path):
	"""
	This should work with every log with minor adjustments
	based on the possible value of lifecycle:transitions
	"""
	log = pm4py.read_xes(path)

	# filter lifecycles
	tracefilter_log = attributes_filter.apply_events(log, ["schedule", "withdraw", "SCHEDULE"],
														   parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "lifecycle:transition", attributes_filter.Parameters.POSITIVE: False})

	for trace in tracefilter_log:
		case_id = trace.attributes["concept:name"]
		trace = trace_add_start_end(trace)
		lifecycle_dict = dict()
		for event in trace:
			event_name = event["concept:name"]
			# if event_name.startswith("W_") and "lifecycle:transition" in event.keys():
			if "lifecycle:transition" in event.keys():
				if event_name in lifecycle_dict.keys():
					lifecycle_dict[event_name] += [event]
				else:
					lifecycle_dict[event_name] = [event]
		for event_name, lifecycle in lifecycle_dict.items():
			# split lifecycle into subcycles
			sublifecycle_list = split_lifecycle(lifecycle)
			for subcycle in sublifecycle_list:
				if len(subcycle) == 1 and subcycle[0]["lifecycle:transition"].lower() == 'complete':
					pass  # nothing to do
				elif subcycle[0]["lifecycle:transition"].lower() != 'start':
					# all subcycles should start with 'start' or 'complete' (in the last case should be the only element)
					raise Exception("Subcycle not starting with 'start' or 'complete': case id %s, event_name: %s" %(case_id, event_name))
				elif subcycle[-1]["lifecycle:transition"].lower() in ['suspend', 'complete', 'ate_abort']:
					# for all complete subcycles the duration is computed and added to the first event
					compute_lifecycle_duration(subcycle)
				else:
					# incomplete subcycles are deleted (they are usually made by single "start" events)
					for event in subcycle:
						event["concept:name"] = "TO_REMOVE"

	tracefilter_log_pos_2 = attributes_filter.apply_events(tracefilter_log, ["TO_REMOVE"],
														   parameters={attributes_filter.Parameters.ATTRIBUTE_KEY: "concept:name", attributes_filter.Parameters.POSITIVE: False})

	return tracefilter_log_pos_2, path


def addRewardSingle(tracefilter_log_pos_2, path):
	max_duration = 3600
	cost_per_second = 0.01
	interest_earning_rate = 0.15
	output_path = path.replace(".xes", "_single_rewards.xes")
	# default_duration = compute_avg_duration(tracefilter_log_pos_2, max_duration)  # instead of taking the avg or median we set an arbitrary default value of 10 min
	default_duration = 600  # 10 minutes
	for trace in tracefilter_log_pos_2:
		objective_event = False  # if O_ACCEPTED is eventually found
		amount = int(trace.attributes[AMOUNT_LABEL]) if AMOUNT_LABEL in trace.attributes.keys() else 0
		for event in trace:
			duration = 0
			reward = 0
			if 'duration' not in event.keys() and event["concept:name"] not in ["START", "END"]:
				duration = default_duration
			if 'duration' in event.keys():
				duration = event['duration']
			reward -= cost_per_second * min(duration, max_duration)
			if event["concept:name"].upper() == "O_ACCEPTED":
				objective_event = True
			if event["concept:name"] == "END" and objective_event:
				reward += interest_earning_rate * amount
			event["kpi:reward"] = round(reward, 2)

	xes_exporter.apply(tracefilter_log_pos_2, output_path)


def addRewardCumulative(tracefilter_log_pos_2, path):
	max_duration = 3600  # 1 hour
	cost_per_second = 0.01
	interest_earning_rate = 0.15
	output_path = path.replace(".xes", "_cumulative_rewards.xes")
	# default_duration = compute_avg_duration(tracefilter_log_pos_2, max_duration)  # instead of taking the avg or median we set an arbitrary default value of 10 min
	default_duration = 600  # 10 minutes
	for trace in tracefilter_log_pos_2:
		objective_event = False  # if O_ACCEPTED is eventually found
		cumulative_reward = 0
		amount = int(trace.attributes[AMOUNT_LABEL]) if AMOUNT_LABEL in trace.attributes.keys() else 0
		for event in trace:
			duration = 0
			reward = 0
			if 'duration' not in event.keys() and event["concept:name"] not in ["START", "END"]:
				duration = default_duration
			if 'duration' in event.keys():
				duration = event['duration']
			reward -= cost_per_second * min(duration, max_duration)
			if event["concept:name"].upper() == "O_ACCEPTED":
				objective_event = True
			if event["concept:name"] == "END" and objective_event:
				reward += interest_earning_rate * amount
			cumulative_reward += reward
			#event["kpi:reward"] = reward
			#event["kpi:cumulative_reward"] = cumulative_reward
			event["kpi:reward"] = round(cumulative_reward, 2)

	xes_exporter.apply(tracefilter_log_pos_2, output_path)



def strip_event_name(path, export_log=True):  # for bpi2017
	# delete spaces
	log = pm4py.read_xes(path)
	for trace in log:
		trace.attributes["amount"] = trace.attributes["RequestedAmount"]
		for event in trace:
			event['concept:name'] = event['concept:name'].replace(" ", "")
	# define output path
	parts = path.split(os.sep)
	output_path = os.sep.join(parts)
	output_path = output_path.replace(".xes", "_strip.xes")
	if export_log:
		xes_exporter.apply(log, output_path)
	return output_path, log

def filter_BPI17_ApplicationType(path, log=None):
	if log is None:
		log = pm4py.read_xes(path)
	log_filtered = filter_trace_attribute(log, attribute_key="ApplicationType", values=["New credit"], retain=True)
	output_path = path.replace(".xes", "_newcredit.xes")
	xes_exporter.apply(log_filtered, output_path)
	return output_path



def utils_main(folder_name):
	# IS23 version
	ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
	real_log_list = ['BPI_2012_log_eng_clean',  # time ordered events and W activities translated in english
					 'BPI_2017_log_strip_newcredit',] # strip extra spaces from activity_labels and filtered by ApplicationType="New credit"
	print('Splitting real logs:')
	for file_name in real_log_list:
		print('\nfile:', file_name)
		path = os.path.join(ROOT, "data", "logs", folder_name, "original", file_name + '.xes')
		splitLog(path, 80)

	log_list = ['BPI_2012_log_eng_clean_training',
				'BPI_2017_log_strip_newcredit_training',
				'event_log_2000_training', 'event_log_rare_2000_training',
				'event_log_4000_training', 'event_log_rare_4000_training',
				'event_log_8000_training', 'event_log_rare_8000_training',
				'event_log_16000_training', 'event_log_rare_16000_training',
				'BPI_2012_log_eng_clean_ForTEST', 'BPI_2017_log_strip_newcredit_ForTEST',    # used to discovery the model used in the BPS evaluation for real log
				'event_log_2000_ForTEST', 'event_log_rare_2000_ForTEST']    # used in the paper only as baselines in the evalution of the policies with the simulator

	for file_name in log_list:
		print('\nfile:', file_name)
		t1 = time.time()
		print("Computing activitiy duration")
		path = os.path.join(ROOT, "data", "logs", folder_name, "original", file_name + '.xes')
		log, path = add_Event_Duration(path)
		# define output folder
		parts = path.split(os.sep)
		parts.remove("original")
		output_path = os.sep.join(parts)
		if file_name.endswith("_training"):
			print("Computing rewards")
			# to training log cumulative reward is added
			addRewardCumulative(log, output_path)
		elif file_name.endswith("_ForTEST"):
			print("Computing rewards")
			# for test log single (non-cumulative) reward is added
			addRewardSingle(log, output_path)
		else:
			raise Exception("Error: file name incorrect:", file_name)
		t2 = time.time()
		print("time:", t2 - t1)


if __name__ == '__main__':
	folder_name = "folder1"
	preprocessing_bpi17 = True
	ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
	if preprocessing_bpi17:
		# prepare bpi17 log
		path = os.path.join(ROOT, "data", "logs", folder_name, "original", "BPI_2017_log.xes")
		path, log = strip_event_name(path, export_log=False)
		filter_BPI17_ApplicationType(path, log)
	utils_main(folder_name)  # add duration and reward to logs