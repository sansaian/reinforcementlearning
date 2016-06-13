from scanext import init
import math
import random

#  author Shalavin Maxim.Sorry I have not experiance on Python.I rewrite my code from java on Python

class QLearn:


    #paraetrs for Q-learning
    alpha = 0.1
    gamma = 0.9

    stateA = 0
    stateB = 1
    stateC = 2
    stateD = 3
    stateE = 4
    stateF = 5

    statesCount = 6

    #Q(s,a)= Q(s,a) + alpha * (R(s,a) + gamma * Max(next state, all actions) - Q(s,a))

    states = (stateA, stateB, stateC, stateD, stateE, stateF)

    #R = np.zeros(6,6) #// reward of matrix
    R = [[0, 0, 0, 0, 0, 0],
         [0, 0, 100, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 100, 0, 0, 0]
         ]
    Q = [[0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0]
         ]
    actionsFromA = [stateB, stateD]
    actionsFromB = [stateA, stateC, stateE]
    actionsFromC = [stateC]
    actionsFromD = [stateA, stateE]
    actionsFromE = [stateB, stateD, stateF]
    actionsFromF = [stateC, stateE]

    actions =[actionsFromA, actionsFromB, actionsFromC,actionsFromD, actionsFromE, actionsFromF]

    stateNames = ["A", "B", "C", "D", "E", "F"];

    def  qlearning(self):
        init(self)
#  filling reward of matrices
    def init(self):
        global R,stateB,stateC,stateF,stateC
        R[stateB][stateC] = 100 # from b to с
        R[stateF][stateC] = 100 # from f to с


    def main(self):

        # long BEGIN = System.currentTimeMillis();
        print("time1")
        q = QLearn()
        q.run()
        q.printResult()
        q.showPolicy()
        print("time2")



# stage for learning alghoritm
    def run (self):
        global statesCount,actions,alpha,gamma
        # 1. Set parameter , and environment reward matrix R
        # 2. Initialize matrix Q as zero matrix
        # 3. For each episode: Select random initial state
        #    Do while not reach goal state o
        #        Select one among all possible actions for the current state o
        #        Using this possible action, consider to go to the next state o
        #        Get maximum Q value of this next state based on all possible actions o
        #        Compute o Set the next state as the current state

        # For each episode
        for g in range(100):
            #// Select random initial state
            state = random.randint(0,self.statesCount-1)
            while (state != self.stateC):

                # goal state
                # Select one among all possible actions for the current state
                actionsFromState = self.actions[state]
                # Selection strategy is random in this example
                index = random.randint(0,len(actionsFromState)-1)
                action = actionsFromState[index]
                #// Action outcome is set to deterministic in this example
                # Transition probability is 1
                nextState = action
                #Using this possible action, consider to go to the next state
                q = self.Q[state] [action]
                maxQvalue = self.maxQ(nextState)
                r = self.R[state][action]
                value = q + self.alpha * (r + self.gamma * maxQvalue - q)
                self.Q [state] [action] = value
                # Set the next state as the current state
                state = nextState


# returns the maximum successful action for the next state where we will move
    def maxQ(self,s):
        global actions,Q
        actionsFromState = self.actions[s]
        # double maxValue = Double.MIN_VALUE; from java i use becouse it is minimal value which not equals 0 but this value approximately equals zero
        maxValue = 4.9*(10**-324)
        for i in actionsFromState:
            #nextState = actionsFromState[i]
            value = self.Q[s][i]
            if (value > maxValue):
                 maxValue = value
        return maxValue

# policy from state
    def policy(self,state):
        global actions, Q
        actionsFromState = self.actions[state]
        maxValue = 4.9 * (10 ** -324)
        policyGotoState = state # default goto self if not found
        for i in actionsFromState:
            value = self.Q[state][i]
            if (value > maxValue):
                maxValue = value
                policyGotoState = i
        return policyGotoState

    def getQ(s,a):
        global Q
        return Q[s],[a]

    def setQ(s,a,value):
        global Q
        Q [s] [a] = value

    def getR(s,a):
        return R[s][a]


    def printResult(self):
        global Q,stateNames
        print("Print result")
        for i in range(len(self.Q)):
            print("out from " + self.stateNames[i] + ":  ")
            for j in range(len(self.Q[i])):
                print(str(self.Q[i][j])+ " ")
            print("|")


# policy is maxQ(states)
    def showPolicy(self):
        global  state,stateNames
        print("\n showPolicy");
        for i in self.states:

            to = self.policy(i)
            print("from "+self.stateNames[i]+" goto "+self.stateNames[to])


if __name__== "__main__":
    q = QLearn()
    q.main()