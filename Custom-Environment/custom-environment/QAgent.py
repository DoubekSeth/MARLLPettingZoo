from collections import Counter
import random
import numpy as np


class QLearningAgent:
    # Note, most of this code is recycled from the UC Berkeley CS188 intro to AI pacman projects. http://ai.berkeley.edu/reinforcement.html
    # This is my solution to the pacman homework problem
    def __init__(self, **args):
        self.env = args['env']
        self.epsilon = args['epsilon']
        self.gamma = args['gamma']
        self.alpha = args['alpha']
        # print(self.epsilon, self.gamma, self.alpha)

        self.qValues = Counter()

        self.weights = np.ones(7)#hardcoded for now, remind me to change later!

        self.featureExtractor = args['features']

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        # Special case: Terminal state
        actions = self.getLegalActions(state)
        #if len(actions) == 0:
            #return 0
        max_action = None
        action_space_size = actions.n
        max_value = float("-inf")
        for action in range(action_space_size):
            currQ = self.getQValue(state, action)
            if currQ > max_value:
                max_value = currQ
                # max_action = action
        return max_value

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        #if len(actions) == 0:
        #    return None
        max_action = None
        max_value = float("-inf")
        action_space_size = actions.n
        for action in range(action_space_size):
            currQ = self.getQValue(state, action)
            print(currQ, action)
            if currQ > max_value:
                max_value = currQ
                max_action = action
        return max_action

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # Special Case: Terminal State
        #if len(legalActions == 0):
        #    return None

        # If true (happens epsilon fraction of times), take random action
        if random.random() < self.epsilon:
            action = random.choice(range(legalActions.n))
        # If not true, choose best outcome
        else:
            action = self.computeActionFromQValues(state)
        # util.raiseNotDefined()

        return action

    def getLegalActions(self, state):
        #Note, might revisit this later when it makes sense
        return self.env.action_space("agent_0")

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        sum = 0
        features = np.append(self.featureExtractor(state['agent'], action, state['observations'], state['env'], r=20, partitions=6),
                             self.distance(target_agent=state['agent'], action=action, observations=state['observations'])/100)#Manually coded, should change
        sum = np.dot(self.weights, features)
        #print(sum, features, self.weights)
        return sum

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        difference = (reward + self.gamma * (self.computeValueFromQValues(nextState))) - self.getQValue(state, action)
        features = np.append(
            self.featureExtractor(state['agent'], action, state['observations'], state['env'], r=20, partitions=6),
            self.distance(target_agent=state['agent'], action=action, observations=state['observations']) / 100)
        self.weights += self.alpha * difference * features

    # Very hacky, will delete later
    def distance(self, target_agent, action, observations):
        """
        Returns the distance between two agents in a given obsevation
        :param agent1: String representation of agent1
        :param agent2: String representation of agent2
        :param observations: Dictionary of observations
        :return: distance
        """
        if(target_agent == "agent_0"):
            agent2 = "agent_1"
        else:
            agent2 = "agent_0"

        displacement = [[-1, -1], [0, -1], [1, -1], [-1, 0], [0, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
        delta = 0.5
        next_x = observations[target_agent]["x"] + displacement[action][0] * delta
        next_y = observations[target_agent]["y"] + displacement[action][1] * delta

        return np.sqrt((next_x - observations[agent2]['x']) ** 2 + (
               next_y - observations[agent2]['y']) ** 2)
