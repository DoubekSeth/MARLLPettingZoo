import functools
import numpy as np

import gymnasium
from gymnasium.spaces import Dict, Discrete, Tuple

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers

NUM_ITERS = 100
SPRING_CONSTANT = 0.5
REPULSION_CONSTANT = 0.5
IDEAL_EDGE_LENGTH = 10


def getCurrentTotalForcesFR(self, agent):
    """
    gets an agent and returns all the forces that act on that agent, namely spring forces and repulsion forces
    :param agent: agent to find the forces for
    :return: total force for an agent as a number
    """
    springForces = calcSpringForces(self, agent)
    repulsionForces = calcRepulsionForces(self, agent)
    totalForces = springForces - repulsionForces
    return np.dot(totalForces, totalForces)**0.5


def calcSpringForces(self, agent):
    """
    Calculates the spring forces for a given agent
    :param agent: agent to find the spring forces for
    :return: a vector representing the sum of all spring forces
    """
    edges = self.state[agent]["edges"]
    springForces = np.zeros(2)

    for edge in edges:
        springForces += calcSpringForce(self, edge)

    return springForces


def calcSpringForce(self, edge):
    """
    Calculate a specific spring force along one edge
    :param edge: edge to calculate the spring force for
    :return: vector representing the force
    """
    source = edge["source"]
    target = edge["target"]

    lengthX = self.state[target]["x"] - self.state[source]["x"]
    lengthY = self.state[target]["y"] - self.state[source]["y"]

    length = (lengthX ** 2 + lengthY ** 2) ** 0.5

    #Avoid division by 0
    if length == 0: return 0

    springForce = SPRING_CONSTANT * np.max([0, length - IDEAL_EDGE_LENGTH])

    springForces = np.array([springForce * (lengthX / length), springForce * (lengthY / length)])

    return springForces


def calcRepulsionForces(self, agent):
    """
    Calculate all the repulsion forces for an agent
    :param self:
    :param agent: agent to find the forces for
    :return: vector representing sum of all repulsion forces
    """
    repulsionForces = np.zeros(2)

    for agentOther in self.agents:
        repulsionForces += calcRepulsionForce(self, agent, agentOther)

    return repulsionForces


def calcRepulsionForce(self, agent, agentOther):
    """
    Calculates a single repulsion force between two nodes
    :param agent: first node
    :param agentOther: second node
    :return: vector representing the repulsion force
    """
    distX = self.state[agent]["x"] - self.state[agentOther]["x"]
    distY = self.state[agent]["y"] - self.state[agentOther]["y"]
    distanceSquared = (distX ** 2 + distY ** 2)
    #Avoid divide by 0
    if distanceSquared == 0 : return 0

    distance = (distX ** 2 + distY ** 2) ** 0.5

    repulsionForce = REPULSION_CONSTANT / distanceSquared

    repulsionForces = np.array([repulsionForce * distX / distance, repulsionForce * distY / distance])

    return repulsionForces


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide variety of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(render_mode=None):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(render_mode=render_mode)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "marllv0.1"}

    def __init__(self, render_mode=None):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.possible_agents = ["agent_" + str(r) for r in range(2)]

        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        self.render_mode = render_mode

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # Implement Later
        return Dict({"sensor": Tuple(Discrete(50), Discrete(50), Discrete(50), Discrete(50), Discrete(50), Discrete(50))})

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(9)

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Current state: Agent0: {}, {} , Agent1: {}, {}".format(
                self.state[self.agents[0]]["x"], self.state[self.agents[0]]["y"],
                self.state[self.agents[1]]["x"], self.state[self.agents[1]]["y"]
            )
        else:
            string = "Game over"
        #print(string)

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        self.agents = self.possible_agents[:]
        self.num_moves = 0
        observations = {agent: {} for agent in self.agents}
        infos = {agent: {} for agent in self.agents}
        #Manually, creating this for now. Hopefully can be passed in
        observations[self.agents[0]]["x"] = -25
        observations[self.agents[0]]["y"] = -25
        observations[self.agents[0]]["edges"] = [{"source": self.agents[0], "target": self.agents[1]}]
        observations[self.agents[1]]["x"] = 25
        observations[self.agents[1]]["y"] = 25
        observations[self.agents[1]]["edges"] = [{"source": self.agents[0], "target": self.agents[1]}]
        self.state = observations

        return observations, infos

    def step(self, actions):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        # Find all old forces, as reward = new forces - old forces
        oldForces = {
            self.agents[i]: getCurrentTotalForcesFR(self, self.agents[i])
            for i in range(len(self.agents))
        }

        terminations = {agent: False for agent in self.agents}

        # current observation is position of state, as well as graph
        observations = {
            self.agents[i]: {"x": self.state[self.agents[i]]["x"], "y": self.state[self.agents[i]]["y"],
                             "edges": self.state[self.agents[i]]["edges"]}
            for i in range(len(self.agents))
        }
        self.state = observations

        #Process each agent's action
        displacement = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
        delta = 0.5
        for agent, action in actions.items():
            currAgent = observations[agent]
            currAgent["x"] = currAgent["x"] + displacement[action][0]*delta
            currAgent["y"] = currAgent["y"] + displacement[action][1]*delta

        self.num_moves += 1
        env_truncation = self.num_moves >= NUM_ITERS
        truncations = {agent: env_truncation for agent in self.agents}

        # Find all new forces, as reward = new forces - old forces
        newForces = {
            self.agents[i]: getCurrentTotalForcesFR(self, self.agents[i])
            for i in range(len(self.agents))
        }

        # get rewards for each agent, might add small negative living reward?
        rewards = {
            self.agents[i]: oldForces[self.agents[i]] - newForces[self.agents[i]]
            for i in range(len(self.agents))
        }

        # typically there won't be any information in the infos, but there must
        # still be an entry for each agent
        infos = {agent: {} for agent in self.agents}

        if env_truncation:
            self.agents = []

        if self.render_mode == "human":
            self.render()
        return observations, rewards, terminations, truncations, infos
