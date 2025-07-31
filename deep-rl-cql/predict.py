from agent import Agent

PATH = './predict/'
MAX_TRACE_LENGTH = 50

# load agent
agent = Agent(path=PATH)

# generate trace
prefix = ['START']

while prefix[-1] != 'END' and len(prefix) < MAX_TRACE_LENGTH:
    prefix.append(agent.get_recommendation(prefix))

print(prefix)
