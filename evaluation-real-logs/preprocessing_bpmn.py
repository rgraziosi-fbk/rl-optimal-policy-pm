import xml.etree.ElementTree as ET
import json


def find_next_acts(A, DFG, bpmn_elements, ACT):
    if len(A) == 1:
        element = A[0]
        if bpmn_elements[element] in ['task', "endEvent"]:
            return [ACT[element]] if bpmn_elements[element] == 'task' else [element]
        else:
            return find_next_acts(DFG[element], DFG, bpmn_elements, ACT)
    else:
        return find_next_acts([A[0]], DFG, bpmn_elements, ACT) + find_next_acts(A[1:], DFG, bpmn_elements, ACT)


def parse_bpmn(file_path):
    ns = {'bpmn': 'http://www.omg.org/spec/BPMN/20100524/MODEL'}
    tree = ET.parse(file_path)
    root = tree.getroot()
    DFG = dict()
    start_node = ''
    end_node = ''

    processes = root.findall('bpmn:process', ns)
    for process in processes:
        elements_TAG = {}
        ACTS = {}
        sequenceFlow_target = {}

        for elem in process:
            tag = elem.tag.split('}')[-1]
            eid = elem.attrib.get('id')
            name = elem.attrib.get('name', eid)

            if tag == 'sequenceFlow':
                sequenceFlow_target[eid] = elem.attrib['targetRef']
                if elem.attrib['sourceRef'] in DFG:
                    DFG[elem.attrib['sourceRef']].append(elem.attrib['targetRef'])
                else:
                    DFG[elem.attrib['sourceRef']] =[elem.attrib['targetRef']]
            else:
                elements_TAG[eid] = tag
                if tag == 'task':
                    ACTS[eid] = name
                    if name == 'START':
                        start_node = eid
                    if name == 'END':
                        end_node = eid

        ###### define a DFG with only the activities
        DFG_only_tasks = dict()
        for key in DFG:
            if elements_TAG[key] == 'task':
                name_act = ACTS[key]
                DFG_only_tasks[name_act] = find_next_acts(DFG[key], DFG, elements_TAG, ACTS)

        #### define a DFG as pm4py object for the alignment of the log
        dfg_pm4py = define_dfg_for_alignment(DFG, start_node, end_node)

        return {"activities": ACTS, "elements_bpmn": elements_TAG, "DFG": DFG, "sequenceFlow_target": sequenceFlow_target,
                "start_node": start_node, "end_node": end_node, "act_node": {ACTS[key]: key for key in ACTS},
                'DFG_only_task': DFG_only_tasks}


def define_dfg_for_alignment(DFG, start_node, end_node):
    sa = {start_node: 1}
    ea = {end_node: 1}
    dfg_pm4py = {}
    for key in DFG:
        for next in DFG[key]:
            dfg_pm4py[(key,next)] = 1
    #pm4py.write_dfg(dfg_pm4py, sa, ea, 'input/BPIC12/BPIC12_DFG.dfg')
    return dfg_pm4py


to_json_file = parse_bpmn("input/BPIC17_parallel/BPIC17_parallel.bpmn")

with open("input/BPIC17_parallel/BPIC17_parallel_DFG.json", "w") as outfile:
    json.dump(to_json_file, outfile, indent=5)