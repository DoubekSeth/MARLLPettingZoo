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
        self.weights = args['weights']
        # print(self.epsilon, self.gamma, self.alpha)

        self.qValues = Counter()
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
        max_action = 4
        max_value = float("-inf")
        action_space_size = actions.n
        for action in range(action_space_size):
            currQ = self.getQValue(state, action)
            #print(currQ, action)
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
            #print("randact")
            action = random.choice(range(legalActions.n))
        # If not true, choose best outcome
        else:
            action = self.computeActionFromQValues(state)
        # util.raiseNotDefined()
        #print("Final weight:", self.weights[6])
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

    def getFeatures(self, state, action):
        feature_funcs = self.featureExtractor
        features = []
        features.append(1/10*feature_funcs[0](state['agent'], action, state['observations']))  # Angle Variance
        features.append(1/2000*feature_funcs[1](state['agent'], action, state['observations'], 100))  # Edge Length Variance
        features.append(1/10*feature_funcs[2](state['agent'], action, state['observations'], state['env'], 4500))  # Dist to all unconnected
        features.append(1/100*feature_funcs[3](state['agent'], action, state['observations'], state['env']))  # Dist to closest Unconnected
        features.append(1/200 * feature_funcs[4](state['agent'], action, state['observations'], state['env'], 100))  # Dist to furthest connected
        features.append(1/100 * feature_funcs[5](state['agent'], action, state['observations'], state['env'], 100))  # Dist to closest connected
        features.append(1/10 * (feature_funcs[6](state['agent'], action, state['observations'], state['env']) + 1)) # Number edge crossings, augmented by 1/10
        features.append(1/10*feature_funcs[-1]())  # Bias func
        features = np.array(features)
        #print(features)
        return features


    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.getFeatures(state, action)
        q_sum = np.dot(self.weights, features)
        #print("weights:", self.weights)
        return q_sum

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        difference = reward + self.gamma * self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        features = self.getFeatures(state, action)
        self.weights += self.alpha * difference * features
        # Need to clip so that weights don't explode :)
        # self.weights = np.clip(self.weights, -50, 50)
        # print(self.weights)


