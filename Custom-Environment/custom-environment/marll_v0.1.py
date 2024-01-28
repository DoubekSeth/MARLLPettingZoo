import math

from env import graph_env
import QAgent

# Numpy
import numpy as np

import matplotlib.pyplot as plt


def distance(agent1, agent2, observations):
    """
    Returns the distance between two agents in a given observation
    :param agent1: String representation of agent1
    :param agent2: String representation of agent2
    :param observations: Dictionary of observations
    :return: distance
    """
    return np.sqrt((observations[agent1]['x'] - observations[agent2]['x']) ** 2 + (
            observations[agent1]['y'] - observations[agent2]['y']) ** 2)


# This function is the same as in https://github.com/kinimesi/cytoscape.js-marll/blob/master/cytoscape-marll.js with minor tweaks
def calcDegree(node, center):
    return math.atan2(node['y'] - center['y'], node['x'] - center['x'])


# This function is the same as in https://github.com/kinimesi/cytoscape.js-marll/blob/master/cytoscape-marll.js with minor tweaks
def calcAngle(node1, center, node2):
    n1dy = node1['y'] - center['y']
    n1dx = node1['x'] - node1['x']
    n2dy = node2['y'] - center['y']
    n2dx = node2['x'] - center['x']
    n1a = math.atan2(n1dy, n1dx)
    n2a = math.atan2(n2dy, n2dx)

    da = abs(n1a - n2a)
    da = da if da < math.pi else 2 * math.pi - da

    return da


def angleVariance(target_agent, action, observations):
    next_x, next_y = takeStep(target_agent, action, observations)
    next_agent = {'x': next_x, 'y': next_y}
    neighbors = sorted(findConnectedNodes(target_agent, observations),
                       key=lambda x: calcDegree(observations[x], next_agent))

    length = len(neighbors)
    variance = 0
    for i in range(length-1):
        variance += (abs(calcAngle(observations[neighbors[i]], next_agent, observations[neighbors[i+1]])) - 2*math.pi/length)**2
    return variance


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
    next_x, next_y = takeStep(target_agent, action, observations)

    vec_counts = np.zeros(partitions)

    for agent in env.agents:
        if agent != target_agent:
            agent_x = observations[agent]['x']
            agent_y = observations[agent]['y']
            # Note: Code below generated by chatgpt then checked and modified by me
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


def distToKNearestNeighborsWithEdgeLength(target_agent, action, observations, env, k, idealEdgeLength):
    """
    Returns the distance between two agents in a given observation
    :param target_agent: String representation of target_agent
    :param action: action taken by the target agent
    :param observations: observations of the state
    :param env: environment of graph
    :param k: number of neighbors
    :return: distance as a vector, with the first being the closest
    """

    next_x, next_y = takeStep(target_agent, action, observations)

    dist_dict = {}
    for agent in env.agents:
        if agent != target_agent:
            dist_dict[agent] = ((np.sqrt((next_x - observations[agent]['x']) ** 2 +
                                         (next_y - observations[agent]['y']) ** 2)) - idealEdgeLength) ** 2

    if k > 1:
        dists = np.sort(np.fromiter(dist_dict.values(), dtype=float))
        return dists[:k]
    else:
        dists = np.fromiter(dist_dict.values(), dtype=float)
        return dists[0]


def distToFarthestConnectedNode(target_agent, action, observations, env, idealEdgeLength):
    """
    Returns the distance between two agents in a given observation
    :param target_agent: String representation of target_agent
    :param action: action taken by the target agent
    :param observations: observations of the state
    :param env: environment of graph
    :return: distance as a vector, with the first being the closest
    """

    next_x, next_y = takeStep(target_agent, action, observations)

    connectedNodes = findConnectedNodes(target_agent, observations)
    largest_dist = 0
    for agent in env.agents:
        if agent != target_agent and agent in connectedNodes:
            dist = ((np.sqrt((next_x - observations[agent]['x']) ** 2 +
                             (next_y - observations[agent]['y']) ** 2)) - idealEdgeLength) ** 2
            if dist > largest_dist:
                largest_dist = dist
    return [largest_dist]


def distToClosestConnectedNode(target_agent, action, observations, env, idealEdgeLength):
    """
    Returns the distance between two agents in a given observation
    :param target_agent: String representation of target_agent
    :param action: action taken by the target agent
    :param observations: observations of the state
    :param env: environment of graph
    :return: distance as a vector, with the first being the closest
    """

    next_x, next_y = takeStep(target_agent, action, observations)

    connectedNodes = findConnectedNodes(target_agent, observations)
    smallest_dist = math.inf
    for agent in env.agents:
        if agent != target_agent and agent in connectedNodes:
            dist = ((np.sqrt((next_x - observations[agent]['x']) ** 2 +
                             (next_y - observations[agent]['y']) ** 2)) - idealEdgeLength) ** 2
            if dist < smallest_dist:
                smallest_dist = dist
    return [smallest_dist]


def findConnectedNodes(target_agent, observations):
    """
    Finds the names of all nodes connected to a node
    :param target_agent: target node
    :param observations: observations of the state space
    :return: list of node names
    """
    edges = observations[target_agent]['edges']
    connectedNodes = []
    for edge in edges:
        if edge['source'] == target_agent:
            connectedNodes.append(edge['target'])
        else:
            connectedNodes.append(edge['source'])
    return connectedNodes


def getOverallForces(observations):
    sumOfForces = 0
    for agent in observations.keys():
        # print(agent, observations[agent]['forces'])
        sumOfForces += observations[agent]['forces']
    return sumOfForces


def takeStep(target_agent, action, observations):
    displacement = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
    delta = 10
    next_x = observations[target_agent]["x"] + displacement[action][0] * delta
    next_y = observations[target_agent]["y"] + displacement[action][1] * delta
    return next_x, next_y


env = graph_env.parallel_env(render_mode="")
# options={"elements": {"nodes": [
#     {"data": {"id": '1'}},
#     {"data": {"id": '0'}},
#     {"data": {"id": '2'}}
# ], "edges": [
#     {"data": {"source": '0', "target": '1', "directed": 'false'}},
#     {"data": {"source": '1', "target": '2', "directed": 'false'}}
# ]},
#     "randomInit": True,
# }
options = {"elements": {
    "nodes": [
        {"data": {"id": 'v1'}},
        {"data": {"id": 'v2'}},
        {"data": {"id": 'v3'}},
        {"data": {"id": 'v4'}},
        {"data": {"id": 'v5'}},
        {"data": {"id": 'v6'}},
        {"data": {"id": 'v0'}}
    ],
    "edges": [
        {"data": {"source": 'v0', "target": 'v1'}},
        {"data": {"source": 'v0', "target": 'v2'}},
        {"data": {"source": 'v0', "target": 'v3'}},
        {"data": {"source": 'v1', "target": 'v4'}},
        {"data": {"source": 'v2', "target": 'v5'}},
        {"data": {"source": 'v3', "target": 'v6'}},
    ]
}, "randomInit": True}
observations, infos = env.reset(options=options)

# Create an agent
learner = QAgent.QLearningAgent(env=env, epsilon=0.15, gamma=0.8, alpha=0.2,
                                features=[distToFarthestConnectedNode, distToClosestConnectedNode, angleVariance,
                                          3])  # Last number is total number of features

NUM_ROLLOUTS = 100
finalSumForces = []
finalWeights = []
for rollout in range(NUM_ROLLOUTS):
    observations, infos = env.reset(options=options)
    while env.agents:
        states = {agent: {'agent': agent, 'observations': observations, 'env': env} for agent in env.agents}
        # actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        actions = {agent: learner.getAction(states[agent]) for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        # print(rewards)
        # print("Action:", actions)
        # print(distance("agent_0", "agent_1", observations))
        # (distance("agent_1", "agent_2", observations))
        # print(observations)
        # print(learner.getWeights())
        nextStates = {agent: {'agent': agent, 'observations': observations, 'env': env} for agent in env.agents}
        for agent in env.agents:
            learner.update(states[agent], actions[agent], nextStates[agent], rewards[agent])
    finalSumForces.append(getOverallForces(observations))
    # finalWeights.append(learner.getWeights())
    # print(observations)
    # print(getOverallForces(observations))
    print(learner.getWeights())
env.close()
# print(finalSumForces)
removeOutliers = np.array(finalSumForces)[np.array(finalSumForces) < 300]

plt.plot(removeOutliers)
plt.show()
# plt.scatter(finalWeights, finalSumForces)
# plt.show()
