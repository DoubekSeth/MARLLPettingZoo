from env import graph_layout
import numpy as np
import numpy.ma as ma

env = graph_layout.parallel_env(render_mode="human")
observations, infos = env.reset()

while env.agents:
    # this is where you would insert your policy
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(observations)
    print(rewards)
    print(terminations)
    print(truncations)
env.close()
