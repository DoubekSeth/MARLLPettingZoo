from env import graph_env

#For graphing
import plotly.graph_objects as go
import networkx as nx

import numpy as np
import numpy.ma as ma

env = graph_env.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    print(actions)
    observations, rewards, terminations, truncations, infos = env.step(actions)
    if not truncations["agent_0"]:
        x0 = observations[env.agents[0]]["x"]
        y0 = observations[env.agents[0]]["x"]
        edges0 = observations[env.agents[0]]["x"]
        x1 = observations[env.agents[1]]["x"]
        y1 = observations[env.agents[1]]["x"]
        edges1 = observations[env.agents[1]]["x"]
env.close()

print(x0)
