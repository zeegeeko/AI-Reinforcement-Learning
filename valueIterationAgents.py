# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        #run the iterations
        for i in range(self.iterations):
            iterVals = util.Counter()
            for state in states:
                #Check Terminal State
                if self.mdp.isTerminal(state):
                    iterVals[state] = 0.0
                else:
                    bestAction = self.computeActionFromValues(state)
                    iterVals[state] = self.computeQValueFromValues(state, bestAction)

            #update the Values before the next iteration
            self.values = iterVals


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        #Just to note:
        #Q(s,a) = \sum_{s'} T(s,a,s')[R(s,a,s') + \gammaV(s')]

        # T is list of (nextState, probability)
        T = self.mdp.getTransitionStatesAndProbs(state, action)
        gamma = self.discount

        return sum([i[1] * (self.mdp.getReward(state,action,i[0]) + gamma * self.values[i[0]])  for i in T])
        #util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        #use dictionary to store bestAction[value] = action
        bestAction = {}
        legalActions = self.mdp.getPossibleActions(state)
        QvalList = []

        #Check for terminal state
        if self.mdp.isTerminal(state):
            return None;

        for a in legalActions:
            val = self.computeQValueFromValues(state, a)
            QvalList.append(val)
            bestAction[val] = a

        return bestAction[max(QvalList)]
        #util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()

        #run the iterations
        for i in range(self.iterations):
            #iterVals = util.Counter()
            #for state in states:

            #Async only updates 1 state per iteration
            #size of states list may be smaller than iteration so need to cycle with mod
            #Check Terminal State
            if self.mdp.isTerminal(states[i % len(states)]):
                #update directly to self.values dictionary
                self.values[states[i % len(states)]] = 0.0
            else:
                bestAction = self.computeActionFromValues(states[i % len(states)])
                #update directly to self.values dictionary
                self.values[states[i % len(states)]] = self.computeQValueFromValues(states[i % len(states)], bestAction)

                #update the Values before the next iteration
                #self.values = iterVals

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        #using dictionary to store predecessors - predecessors[state] = pred states
        states = self.mdp.getStates()
        predecessors = self.getPredecessors(states)

        #Initialize an empty priority queue.
        pqueue = util.PriorityQueue()

        #For each non-terminal state s, do:
        for s in states:
            #find absolute difference of state in self.values and highest Q value of
            #all actions of s
            if self.mdp.isTerminal(s):
                continue
            diff = abs(self.values[s] - self.maxQvalue(s))
            #Push s into the priority queue with priority -diff
            pqueue.push(s,-diff)

        for i in range(self.iterations):
            #terminate if priority queue is emtpy
            if pqueue.isEmpty():
                break

            #pop state off queue
            state = pqueue.pop()

            if not self.mdp.isTerminal(state):
                self.values[state] = self.maxQvalue(state)

            #for each predecessor p of state
            for p in predecessors[state]:
                #print("state: ",state)
                #print("pre: ", p)

                #get absolute difference of state p in self.values and highest Qval of p
                diff = abs(self.values[p] - self.maxQvalue(p))

                #if diff > theta update push p into priorty queue
                if diff > self.theta:
                    #use update method instead
                    pqueue.update(p,-diff)


    #returns a dictionary of states and predecessors
    def getPredecessors(self, states):
        predecessors = {}

        #Compute predecessors of all states.
        #1. iterate to each state
        #for s in states:
            #if self.mdp.isTerminal(s):
            #    continue
        #    predecessors[s] = set()
            #2. for each state (s), iterate through list of states (i) and check
            # to see which states lead to state (s) with probability > 0
        #    for i in states:
                #if self.mdp.isTerminal(i):
                #    continue
                #3. Getting list of legal actions for each state (i)
        #        for a in self.mdp.getPossibleActions(i):
        #            Tval = self.mdp.getTransitionStatesAndProbs(i,a)
                    #4. Evaluate each of the nextStates from Tval see if they lead
                    # to state (s) with probability > 0. Add to predecessors dictionary
        #            for t in Tval:
        #                if t[0] == s and t[1] > 0.0:
        #                    predecessors[s].add(t[0])
        for s in states:
            predecessors[s] = set()
        for s in states:
            for a in self.mdp.getPossibleActions(s):
                for Tval in self.mdp.getTransitionStatesAndProbs(s, a):
                    s2 = Tval[0]
                    if Tval[1] > 0:
                        predecessors[s2].add(s)
        return predecessors

    def maxQvalue(self, state):
        actions = self.mdp.getPossibleActions(state)
        gamma = self.discount
        Qvals = []

        #no legal actions from this state
        if not actions:
            return 0.0

        for a in actions:
            #Tval = self.mdp.getTransitionStatesAndProbs(state, a)
            #summation = 0.0
            #Calculate Q values without updating self.values.
            #for t in Tval:
            #    summation += t[1] * (self.mdp.getReward(state,a,t[0]) + gamma * self.values[t[0]])
            Qvals.append(self.computeQValueFromValues(state, a))

        return max(Qvals)
