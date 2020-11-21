# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import pdb
import math
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        print(scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        oldFood =currentGameState.getFood()
        newFood = successorGameState.getFood()
        oldGhostPos=currentGameState.getGhostPositions()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        count=successorGameState.getScore()
        nearestFood=nearestParticle(successorGameState)
        #print(currentGameState.getNumFood()-successorGameState.getNumFood())
        minDist=10000
        for ghostPos in oldGhostPos:
            minDist=min(minDist,manhattanDistance(ghostPos,newPos))        
        villComp=(2/math.log(minDist+0.1,10))
        foodComp=nearestFood
        if(currentGameState.getNumFood()-successorGameState.getNumFood() > 0):
            foodComp-=20
        cost=count-foodComp-villComp
        print(cost,count,foodComp,villComp,nearestFood)
        return cost

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        index=1
        #agents=range(0,gameState.getNumAgents())
        actions=gameState.getLegalActions(0)
        dep=self.depth
        score=float("-inf"),actions[-1]
        for action in actions:
            minmaxRecurs=self.minmaxRecursor(index,gameState.getNumAgents(),gameState.generateSuccessor(0, action),dep-1)
            if score[0]<minmaxRecurs:
                score=max(score[0],minmaxRecurs) ,action
        print(score)
        return score[1]
        
    def minmaxRecursor(self,index,agents,gameState,depth):
        index=index%agents
        if (depth==0 and index==0) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)        
        actions=gameState.getLegalActions(index)
        if(index==0):
            score=float("-inf")
            for action in actions:
                score=max(score,self.minmaxRecursor(index+1,agents,gameState.generateSuccessor(index, action),depth-1))
            print(score)
            return score
        else:
            score=float("inf")
            for action in actions:
                    score=min(score,self.minmaxRecursor(index+1,agents,gameState.generateSuccessor(index, action),depth))
            print(score)
            return score


class AlphaBetaAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        "*** YOUR CODE HERE ***"
        index=0
        actions=gameState.getLegalActions(index)
        dep=self.depth
        score=float("-inf"),actions[-1]
        alpha=float("-inf")
        beta=float("inf")
        
        for action in actions:
            alpharecurse=self.abrecurse(index+1,gameState.getNumAgents(),gameState.generateSuccessor(index, action),dep-1,alpha,beta)
            if score[0]<alpharecurse:
                score= alpharecurse,action
            #print(alpha,beta)
            alpha=max(alpha,score[0])
        return score[1]
        util.raiseNotDefined()
    
    def abrecurse(self,index,agents,gameState,depth,alpha,beta):
        index=index%agents
        
        if (depth==0 and index==0) or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)        
        
        actions=gameState.getLegalActions(index)
        
        if(index==0):
            score=float("-inf")
            for action in actions:
                abpoints=self.abrecurse(index+1,agents,gameState.generateSuccessor(index, action),depth-1,alpha,beta)
                score=max(score,abpoints)
                if(score>beta):
                    return score
                alpha=max(alpha,score)
                #pdb.set_trace()
            return score
        
        else:
            score=float("inf")
        
            if(index==1):
                for action in actions:
                    abpoints=self.abrecurse(index+1,agents,gameState.generateSuccessor(index, action),depth,alpha,beta)
                    score=min(score,abpoints)
                    beta=min(beta,score)
                    if alpha>score:
                        return score
                    #pdb.set_trace()
                return score
            
            else:
                for action in actions:
                    abpoints=self.abrecurse(index+1,agents,gameState.generateSuccessor(index, action),depth,alpha,beta)
                    score=min(score,abpoints)
                    beta=min(beta,score)
                    if alpha>score:
                        return score
                    #pdb.set_trace()
                return score


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        index=1
        actions=gameState.getLegalActions(0)
        dep=self.depth
        score=float("-inf"),actions[-1]
        for action in actions:
            recursor=self.expectimax(index,gameState.getNumAgents(),gameState.generateSuccessor(0, action),dep-1)
            if score[0]<recursor:
                score=recursor,action
        print(score)
        return score[1]
        

    def expectimax(self,index,agents,gameState,dep):
        index=index%agents
        if ((index==0 and dep==0)  or gameState.isWin() or gameState.isLose()):
            print((index==0 & dep==0),dep,gameState.isWin(),gameState.isLose())
            return self.evaluationFunction(gameState)
        
        actions=gameState.getLegalActions(index)
        if(index==0):
            score=float("-inf")
            for action in actions:
                score=max(score,self.expectimax(index+1,agents,gameState.generateSuccessor(index, action),dep-1))
            return score
        else:
            score=0
            for action in actions:
                score+=self.expectimax(index+1,agents,gameState.generateSuccessor(index, action),dep)
            return score

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


def nearestParticle(gameState):
    pacPos=gameState.getPacmanPosition()
    food=gameState.getFood()
    if(food[pacPos[0]][pacPos[1]]):
        #pdb.set_trace()
        return -20
    walls=gameState.getWalls()
    queue=[]
    myset = set() 
    queue.append(pacPos)
    queue.append(None)
    layer=0
    while(len(queue)!=1):
        if(queue[0]==None):
                layer+=1
                queue.append(None)
                queue.pop(0)
        else:
            pacpos=queue.pop(0)
            pacposX=pacpos[0]
            pacposY=pacpos[1]
            if(food[pacposX][pacposY]):
                return layer
            hashtemp=str(pacposX)+""+str(pacposY)
            if(hashtemp in myset):
                continue
            else:
                myset.add(hashtemp)
            if(walls[pacposX+1][pacposY]==False):
                queue.append([pacposX+1,pacposY])
            if(walls[pacposX-1][pacposY]==False):
                queue.append([pacposX-1,pacposY])
            if(walls[pacposX][pacposY+1]==False):
                queue.append([pacposX,pacposY+1])
            if(walls[pacposX][pacposY-1]==False):
                queue.append([pacposX,pacposY-1])
    return -30
               
