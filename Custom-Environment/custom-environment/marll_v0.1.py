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
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards)
env.close()
