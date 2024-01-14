from env import graph_env
import QAgent

# For graphing
import plotly.graph_objects as go
import networkx as nx

# Numpy
import numpy as np


def distance(agent1, agent2, observations):
    """
    Returns the distance between two agents in a given obsevation
    :param agent1: String representation of agent1
    :param agent2: String representation of agent2
    :param observations: Dictionary of observations
    :return: distance
    """
    return np.sqrt((observations[agent1]['x'] - observations[agent2]['x']) ** 2 + (
            observations[agent1]['y'] - observations[agent2]['y']) ** 2)


def partition_circle(target_agent, observations, env, r, partitions):
    """
    This returns a vector that represents whether an agent is within radius r of the target agent.
    If it is within the radius, then the vector represents the count that corresponds to each partition.
    Ex: (1, 0, 2, 1) means 1 object is within radius r and in first quadrant, 0 objects in second quadrant...
    :param target_agent: target agent centered around
    :param observations: observations for a given step
    :param env: current environment
    :param r: radius of the circle
    :param partitions: number of partitions
    :return: vector of counts
    """
    vec_counts = np.zeros(partitions)
    target_x = observations[target_agent]['x']
    target_y = observations[target_agent]['y']

    for agent in env.agents:
        if agent != target_agent:
            agent_x = observations[agent]['x']
            agent_y = observations[agent]['y']
            # Note: Code below generated by chatgpt then checked modified by me
            # Convert Cartesian coordinates to polar coordinates
            radius, theta = distance(target_agent, agent, observations), np.arctan2((agent_y - target_y),
                                                                                    (agent_x - target_x))

            # Normalize theta to be in the range [0, 2*pi)
            theta = (theta + 2 * np.pi) % (2 * np.pi)
            # Calculate the sector size for six partitions
            sector_size = 2 * np.pi / partitions

            # Determine the partition index
            partition_index = np.floor(theta / sector_size).astype(int)

            # Only add if within distance
            if radius <= r:
                vec_counts[partition_index] += 1
    return vec_counts


def partition_circle_act(target_agent, action, observations, env, r, partitions):
    """
    This returns a vector that represents whether an agent is within radius r of the target agent.
    If it is within the radius, then the vector represents the count that corresponds to each partition.
    Ex: (1, 0, 2, 1) means 1 object is within radius r and in first quadrant, 0 objects in second quadrant...
    :param target_agent: target agent centered around
    :param action: the action that the target agent will take
    :param observations: observations for a given step
    :param env: current environment
    :param r: radius of the circle
    :param partitions: number of partitions
    :return: vector of counts
    """
    # Do the action step!
    # Process each agent's action
    displacement = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
    delta = 0.5
    next_x = observations[target_agent]["x"] + displacement[action][0] * delta
    next_y = observations[target_agent]["y"] + displacement[action][1] * delta

    vec_counts = np.zeros(partitions)

    for agent in env.agents:
        if agent != target_agent:
            agent_x = observations[agent]['x']
            agent_y = observations[agent]['y']
            # Note: Code below generated by chatgpt then checked modified by me
            # Convert Cartesian coordinates to polar coordinates
            radius, theta = distance(target_agent, agent, observations), np.arctan2((agent_y - next_y),
                                                                                    (agent_x - next_x))

            # Normalize theta to be in the range [0, 2*pi)
            theta = (theta + 2 * np.pi) % (2 * np.pi)
            # Calculate the sector size for six partitions
            sector_size = 2 * np.pi / partitions

            # Determine the partition index
            partition_index = np.floor(theta / sector_size).astype(int)

            # Only add if within distance
            if radius <= r:
                vec_counts[partition_index] += 1
    return vec_counts


env = graph_env.parallel_env(render_mode="human")
observations, infos = env.reset(options={"elements": {"nodes": [
    {"data": {"id": '1'}},
    {"data": {"id": '0'}}
], "edges": [
    {"data": {"source": '0', "target": '1', "directed": 'false'}}
    ]},
    "randomInit": True,
})

# Create an agent
learner = QAgent.QLearningAgent(env=env, epsilon=0.05, gamma=0.8, alpha=0.2, features=partition_circle_act)

while env.agents:
    states = {agent: {'agent': agent, 'observations': observations, 'env': env} for agent in env.agents}
    # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    actions = {agent: learner.getAction(states[agent]) for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(rewards)
    print("Action:", actions)
    print(distance("agent_0", "agent_1", observations))
    nextStates = {agent: {'agent': agent, 'observations': observations, 'env': env} for agent in env.agents}
    for agent in env.agents:
        learner.update(states[agent], actions[agent], nextStates[agent], rewards[agent])
env.close()
