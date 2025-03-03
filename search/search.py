# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    from util import Stack  # Stack ensures LIFO (last-in, first-out) behavior

    my_stack = Stack()  # Stack stores (node, path) pairs (tuples of size two)
    visited = set()  # Tracks visited states, a set of coordinates. 

    my_stack.push((problem.getStartState(), []))  # Start state with an empty path

    while not my_stack.isEmpty():
        node, path = my_stack.pop()  # Pop the most recent state, store its coordinates (node), and the path to this coordinate (path)

        if problem.isGoalState(node):  # Check if we have reached pacmans final goal. 
            return path

        if node not in visited:
            visited.add(node)  # Mark grid location as visited
            for nextNode, action, _ in problem.getSuccessors(node): # Recall successors returns both state (coordinates), and actions (path), see searchAgents.py
                if nextNode not in visited:
                    my_stack.push((nextNode, path + [action]))  # Add the new node, along with the entire path to get there. 
                    # If we did not add the previous path, we would lose the previous nodes path, and only ever be looking ahead by one action. 
    return []  # Return empty if no solution found

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    from util import Queue # we need Queue for breadth first (FIFO), this is the only key difference from DFS. 

    # Queue format ((node),[path to this node]) #
    my_q = Queue()

    visited = set()  # Tracks visited states
 
     # Check if initial state is the end goal.
    # Simply return an empty list in this case, state is already valid. 
    if problem.isGoalState(problem.getStartState()):
        return []

    # Push initial state, the initial path is an empty list #
    my_q.push((problem.getStartState(),[]))

    while not my_q.isEmpty():
        node, path = my_q.pop()  # Dequeue the oldest state

        if problem.isGoalState(node):  # Goal check
            return path

        if node not in visited:
            visited.add(node)  # Mark as visited
            for nextNode, action, _ in problem.getSuccessors(node):
                if nextNode not in visited:
                    my_q.push((nextNode, path + [action]))  # Add new path to queue

    return []  # Return empty if no solution found

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    from util import PriorityQueue
    from util import PriorityQueueWithFunction # we need a priority queue (cost based) for UCS.
    # Remember: priority queu always returns the next cheapest node...see util.py. 
      
    # This creates a priority queue where each item’s priority is determined by a function.
    # This in line function always extract the 3rd element of the tuple, and that will be the priority (the third element is cost)
    my_pq = PriorityQueueWithFunction(lambda state: state[2])

    # Create an empty dictionary for visited nodes
    # It is not a set this time, because we need the node, AND its cost. 
    # A key,value pair
    visited = {}

    path = [] # Each node knows the path to its coordinate
    cost = 0; # Each node has an associated cost, the initial cost is 0. 

     # Check if initial state is the end goal.
    # Simply return an empty list in this case, state is already valid. 
    if problem.isGoalState(problem.getStartState()):
        return []

    # Push initial state, a tuple
    # Queue format (((node),[path to this node], cost of action),priority) 
    # We push 4 pieces of info, a tuple (node, path to node, cost)
    my_pq.push((problem.getStartState(), path, cost))



    while not my_pq.isEmpty():
        node, path, cost = my_pq.pop()  # Pops the cheapest node by default (That is priority queues job). 

        if problem.isGoalState(node):  # Goal check
            return path

        # We may encounter the same node twice, but under cheaper circumstances. I.E we hit the goal, but faster than last time.
        if node not in visited or cost < visited[node]:
            visited[node] = cost  # Mark as visited with its cost, OR update the cost with the new cheaper version. 
            for nextNode, action, nextCost, in problem.getSuccessors(node):
                my_pq.push((nextNode, path + [action], cost + nextCost))  # Push next state
                    # We don't just update the path this time, we must also update the cost for each subsequent node. 

    return []  # Return empty if no solution found


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost (g + h) first."""
    from util import PriorityQueueWithFunction

    # Define the priority function: f(n) = g(n) + h(n)
    priorityFunction = lambda item: item[2] + heuristic(item[0], problem)  # f(n) = g(n) + h(n)
    # Cost + heuristic(node, problem)

    # Initialize the priority queue with this function
    frontier = PriorityQueueWithFunction(priorityFunction)
    visited = {}  # Dictionary to track best cost to reach each state

    start_state = problem.getStartState()
    frontier.push((start_state, [], 0))  # (state, path, g(n))

    while not frontier.isEmpty():
        state, path, cost = frontier.pop()  # Get node with lowest f(n)

        if problem.isGoalState(state):
            return path

        if state in visited and visited[state] <= cost:
            continue

        visited[state] = cost  # Mark this state with the best cost

        for successor, action, step_cost in problem.getSuccessors(state):
            new_cost = cost + step_cost  # g(n)
            frontier.push((successor, path + [action], new_cost))  # No need to specify priority explicitly!

    return []  # No solution found




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
