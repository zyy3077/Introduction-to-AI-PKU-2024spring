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
from math import sqrt, log
from game import Agent
from copy import deepcopy
from FruitModel import modeldict
import os


def predfigure(agent, gameState):
    print_label = {0:'negative', 1:'positive'}
    if gameState.data.layout.predy is None: 
        predy = gameState.data.layout.figureidx.copy()
        for (x, y) in gameState.data.food.asList():
            data = gameState.data.layout.data[gameState.data.layout.dataidx[x][y]-1]
            if gameState.data.layout.task == 'classifier': 
                predy[x][y] = agent.model(data[0])
                print('坐标: {} {} 待分类句子: {} 正确结果: {} 预测结果: {}'.format(x, y, data[1], print_label[gameState.data.layout.figureidx[x][y]], print_label[predy[x][y]]))
            else:
                pred_ans = agent.model(data[0])
                predy[x][y] = (pred_ans == data[1])
                print('---------------\n坐标: {} {}\n问题: {}\n正确结果: {}\n输出结果: {}'.format(x, y, data[0], data[1], pred_ans))
        gameState.data.layout.predy = predy
        os.system("pause")
    else:
        predy = gameState.data.layout.predy
    return predy, gameState


def getmodel(name):
    ret = modeldict.get(name, None)
    assert ret is not None, f"model {name} is not implemented in MnistModel/modeldict, available models are {list(modeldict.keys())}"
    return ret


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', model: str="Null"):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.model = getmodel(model)()

    def getVisAction(self, gameState):
        pass

    def getAction(self, gameState):
        predy, gameState = predfigure(self, gameState)
        gameState = deepcopy(gameState)
        gameState.data.layout.y = gameState.data.layout.predy
        action = self.getVisAction(gameState)
        return action, predy

from util import Queue
import pdb
class ReflexAgent(MultiAgentSearchAgent):
    def bfsstate(self, gameState):
        gameState = deepcopy(gameState)
        visited = set()
        queue = Queue()
        queue.push((gameState,1))
        visited.add(gameState.getPacmanPosition())
        foods = gameState.getPosFood().asList()
        while not queue.isEmpty():
            cur, curdist = queue.pop()
            ndist = curdist + 1
            for act in cur.getLegalActions():
                next = cur.generatePacmanSuccessor(act)
                npos = next.getPacmanPosition()
                if npos not in visited:
                    if npos in foods:
                        return ndist
                    queue.push((next, ndist))
                    visited.add(npos)
        return 1000


    def ReflexevaluationFunction(self, currentGameState, action):
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getPosFood().asList()
        mindist = 0
        if len(newFood) > 0:
            mindist = self.bfsstate(successorGameState)
        return successorGameState.getScore() - 1e-3*mindist

    def getVisAction(self, gameState):
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()
        if len(legalMoves) > 1:
            legalMoves = [action for action in legalMoves if action != "Stop"]
        # print(legalMoves)
        # Choose one of the best actions
        scores = [self.ReflexevaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)
        # pdb.set_trace()
        return legalMoves[chosenIndex]
